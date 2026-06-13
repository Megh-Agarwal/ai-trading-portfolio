"""Abstract base class for all LLM trading agents."""
from __future__ import annotations

import datetime
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import anthropic
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cache import cached_call as _default_cached_call
from db.models import AgentCall, Signal

logger = logging.getLogger(__name__)

_REGIME_TO_FLOAT = {"risk_on": 1.0, "neutral": 0.0, "risk_off": -1.0}
_RATE_OUTLOOK_TO_FLOAT = {"rising": 1.0, "stable": 0.0, "falling": -1.0}


class BaseAgent(ABC):
    """Abstract base for all three LLM trading agents.

    Subclasses must set:
        agent_name: str          — matches key in agents.yaml and agent_calls table
        _schema_class: type      — Pydantic model class for validate_output

    Subclasses must implement:
        prepare_input(date, db) -> dict
        _write_signals(date, validated, call_id, db) -> None
    """

    agent_name: str
    _schema_class: type

    def __init__(
        self,
        model_string: str,
        prompt_template_path: Path | str,
        cache: Callable | None = None,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._model = model_string
        self._prompt_path = Path(prompt_template_path)
        if not self._prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self._prompt_path}")
        self._prompt = self._prompt_path.read_text(encoding="utf-8")
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Injected for testing; production uses the module-level cached_call.
        self._cached_call = cache if cache is not None else _default_cached_call

    @abstractmethod
    def prepare_input(self, date: datetime.date, db: Engine) -> dict:
        """Collect and structure all input data for the LLM call.

        Args:
            date: The rebalance date being processed.
            db: SQLAlchemy engine for querying historical data.

        Returns:
            JSON-serialisable dict passed to the LLM as user message content.
        """

    def validate_output(self, response: dict) -> dict:
        """Parse and validate the LLM response against the agent's Pydantic schema.

        Args:
            response: Raw response dict from cached_call (Anthropic API shape).

        Returns:
            Validated dict matching the agent's schema.

        Raises:
            ValueError: JSON extraction fails or schema validation fails.
        """
        content = response.get("content", [])
        text = ""
        for block in content:
            # Handle both raw dicts (from cache) and SDK objects (live call).
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block["text"]
                    break
            elif hasattr(block, "type") and block.type == "text":
                text = block.text
                break

        if not text:
            raise ValueError("LLM response contained no text content block")

        # Strip markdown code fences if the model wraps JSON in ```json ... ```.
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            inner = lines[1:] if len(lines) > 1 else lines
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            stripped = "\n".join(inner)

        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM output is not valid JSON: {exc}\n\nRaw text:\n{text}"
            ) from exc

        validated = self._schema_class.model_validate(raw)
        return validated.model_dump()

    @abstractmethod
    def _write_signals(
        self,
        date: datetime.date,
        validated: dict,
        call_id: int | None,
        db: Engine,
    ) -> None:
        """Write validated signal rows to the signals table.

        Args:
            date: Rebalance date.
            validated: Output of validate_output().
            call_id: FK to agent_calls row, or None if DB logging is disabled.
            db: SQLAlchemy engine.
        """

    def run(self, date: datetime.date, db: Engine) -> dict:
        """Full pipeline: prepare_input → cached LLM call → validate → write signals.

        Args:
            date: Rebalance date to process.
            db: SQLAlchemy engine.

        Returns:
            Validated signal dict.
        """
        input_data = self.prepare_input(date, db)

        def _call_fn() -> dict:
            # anthropic.Anthropic() is only reached on a cache miss inside cached_call.
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._prompt,
                messages=[{"role": "user", "content": json.dumps(input_data, default=str)}],
            )
            return message.model_dump()

        response = self._cached_call(
            self._model,
            self._prompt,
            input_data,
            _call_fn,
            agent_name=self.agent_name,
            engine=db,
        )

        validated = self.validate_output(response)
        call_id = self._last_call_id(db)
        self._write_signals(date, validated, call_id, db)
        return validated

    # ------------------------------------------------------------------
    # Shared signal-writing helpers for subclasses
    # ------------------------------------------------------------------

    def _insert_signals(self, rows: list[Signal], db: Engine) -> None:
        with Session(db) as session:
            session.add_all(rows)
            session.commit()

    def _last_call_id(self, db: Engine) -> int | None:
        """Return call_id of the most recent agent_calls row for this agent."""
        try:
            with Session(db) as session:
                row = (
                    session.query(AgentCall)
                    .filter(AgentCall.agent_name == self.agent_name)
                    .order_by(AgentCall.call_id.desc())
                    .first()
                )
                return row.call_id if row else None
        except Exception:
            return None
