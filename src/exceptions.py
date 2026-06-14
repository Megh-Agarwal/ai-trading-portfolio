"""Project-wide custom exceptions."""
from __future__ import annotations


class TruncationError(RuntimeError):
    """Raised when an LLM response was cut off at the max_tokens limit.

    Silent truncation drops required fields (e.g. evidence) and causes
    downstream ValidationError. Fail loud here instead.
    """
