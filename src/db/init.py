from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine

from db.models import Base

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "state.db"


def init_db(db_path: Path | str = _DEFAULT_DB_PATH) -> None:
    """Create all tables at db_path if they do not already exist (idempotent)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    logger.info("Database initialized at %s", db_path)
