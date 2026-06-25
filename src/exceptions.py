"""Project-wide custom exceptions."""

from __future__ import annotations


class TruncationError(RuntimeError):
    """Raised when an LLM response was cut off at the max_tokens limit.

    Silent truncation drops required fields (e.g. evidence) and causes
    downstream ValidationError. Fail loud here instead.
    """


class NegativeCashError(RuntimeError):
    """Raised when fills drive the CASH position below zero.

    This should never happen if validate_orders_affordable passed. If it
    fires it means the affordability check was bypassed or the fill math
    has a bug — treat as a hard programming error, not a recoverable state.
    """
