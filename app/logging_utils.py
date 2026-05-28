"""Small logging helpers shared by application modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

LogValue: TypeAlias = str | int | float | bool | None
LogFields: TypeAlias = Mapping[str, LogValue]


def log_extra(event: str, **fields: LogValue) -> dict[str, LogValue]:
    """Build structured logging fields without leaking unset values."""

    return {"event": event, **{key: value for key, value in fields.items() if value is not None}}
