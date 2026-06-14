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
        _tool: dict              — Anthropic tool definition (name, description, input_schema)

    Subclasses must implement:
        prepare_input(date, db) -> dict
        _write_signals(date, validated, call_id, db) -> None
    """

    agent_name: str
    _schema_class: type
    _tool: dict

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
        """Collect and structure all input data for the LLM call."""

    def validate_output(self, response: dict) -> dict:
        """Extract and validate the LLM tool_use response against the agent's Pydantic schema.

        Args:
            response: Raw response dict from cached_call (Anthropic API shape).

        Returns:
            Validated dict matching the agent's schema.

        Raises:
            ValueError: No tool_use block found or schema validation fails.
        """
        content = response.get("content", [])
        tool_input = None
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    tool_input = block.get("input")
                    break
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_input = block.input
                break

        if tool_input is None:
            raise ValueError("LLM response contained no tool_use content block")

        validated = self._schema_class.model_validate(tool_input)
        return validated.model_dump()

    @abstractmethod
    def _write_signals(
        self,
        date: datetime.date,
        validated: dict,
        call_id: int | None,
        db: Engine,
    ) -> None:
        """Write validated signal rows to the signals table."""

    def run(self, date: datetime.date, db: Engine) -> dict:
        """Full pipeline: prepare_input → cached LLM tool call → validate → write signals."""
        input_data = self.prepare_input(date, db)

        def _call_fn() -> dict:
            client = anthropic.Anthropic()
            message = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._prompt,
                tools=[self._tool],
                tool_choice={"type": "tool", "name": self._tool["name"]},
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
            tool=self._tool,
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
