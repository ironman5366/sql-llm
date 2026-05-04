from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any


LOGGER = logging.getLogger("sql_llm")

EventSink = Callable[[dict[str, Any]], None]
_event_sink: contextvars.ContextVar[EventSink | None] = contextvars.ContextVar("sql_llm_event_sink", default=None)

_PHASE_COLORS = {
    "boot": "37",
    "adapter": "36",
    "sampling": "34",
    "training": "35",
    "checkpoint": "33",
    "mutation": "32",
    "error": "31",
}
_DEFAULT_COLOR = "37"
_RESET = "\033[0m"


def _color_enabled() -> bool:
    override = os.environ.get("SQL_LLM_COLOR")
    if override is not None:
        return override.lower() in {"1", "true", "yes", "always"}
    return sys.stderr.isatty()


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        phase = getattr(record, "sql_llm_phase", None) or _PHASE_FROM_LEVEL.get(record.levelno, "log")
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        tag = f"{phase:<10}"
        if self.use_color:
            color = _PHASE_COLORS.get(phase, _DEFAULT_COLOR)
            tag = f"\033[1;{color}m{tag}{_RESET}"
            timestamp = f"\033[90m{timestamp}{_RESET}"
        prefix = f"{timestamp}  {tag}  "
        lines = message.splitlines() or [""]
        return "\n".join(f"{prefix if i == 0 else ' ' * _visible_len(prefix)}{line}" for i, line in enumerate(lines))


_PHASE_FROM_LEVEL = {
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}


def _visible_len(text: str) -> int:
    out = 0
    in_escape = False
    for ch in text:
        if in_escape:
            if ch == "m":
                in_escape = False
            continue
        if ch == "\033":
            in_escape = True
            continue
        out += 1
    return out


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_ConsoleFormatter(use_color=_color_enabled()))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Quiet noisy third-party loggers so the scheme's logs stay readable in a demo.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "transformers", "datasets", "accelerate"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_event_sink(sink: EventSink | None) -> contextvars.Token[EventSink | None]:
    return _event_sink.set(sink)


def reset_event_sink(token: contextvars.Token[EventSink | None]) -> None:
    _event_sink.reset(token)


def emit_event(event: dict[str, Any]) -> None:
    sink = _event_sink.get()
    if sink is not None:
        sink(dict(event))


def log_section(phase: str, title: str, *, char: str = "─", width: int = 72) -> None:
    bar = char * max(4, width - len(title) - 3)
    LOGGER.info("%s %s", title, bar, extra={"sql_llm_phase": phase})


def emit_progress(
    phase: str,
    message: str,
    *,
    percent: float | None = None,
    step: int | None = None,
    total: int | None = None,
    **fields: Any,
) -> None:
    event: dict[str, Any] = {
        "event": "progress",
        "phase": phase,
        "message": message,
        **fields,
    }
    if percent is not None:
        event["percent"] = max(0.0, min(100.0, float(percent)))
    if step is not None:
        event["step"] = step
    if total is not None:
        event["total"] = total

    LOGGER.info(_progress_line(event), extra={"sql_llm_phase": phase})
    emit_event(event)


def log_query(kind: str, label: str, query: str, *, max_chars: int = 4000, **fields: Any) -> None:
    phase = _phase_for_kind(kind)
    LOGGER.info(
        "▶ %s · %s\n%s",
        kind,
        label,
        _indent(_truncate(query, max_chars)),
        extra={"sql_llm_phase": phase},
    )
    emit_event(
        {
            "event": f"{kind}_query",
            "label": label,
            "query": _truncate(query, max_chars),
            **fields,
        }
    )


def log_tokens(kind: str, label: str, text: str, token_ids: list[int], *, max_chars: int = 4000) -> None:
    phase = _phase_for_kind(kind)
    LOGGER.info(
        "◀ %s · %s · %d tokens\n%s",
        kind,
        label,
        len(token_ids),
        _indent(_truncate(text, max_chars)),
        extra={"sql_llm_phase": phase},
    )
    emit_event(
        {
            "event": f"{kind}_tokens",
            "label": label,
            "token_count": len(token_ids),
            "token_ids": token_ids[:128],
            "text": _truncate(text, max_chars),
        }
    )


def log_json(kind: str, label: str, payload: Any, *, max_chars: int = 4000) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)
    LOGGER.info(
        "%s · %s\n%s",
        kind,
        label,
        _indent(_truncate(rendered, max_chars)),
        extra={"sql_llm_phase": _phase_for_kind(kind)},
    )
    emit_event({"event": kind, "label": label, "payload": _truncate(rendered, max_chars)})


def _progress_line(event: dict[str, Any]) -> str:
    parts = [event["message"]]
    if "step" in event and "total" in event:
        parts.append(f"[{event['step']}/{event['total']}]")
    if "percent" in event:
        parts.append(f"({event['percent']:.1f}%)")
    return " ".join(parts)


def _phase_for_kind(kind: str) -> str:
    if kind.startswith("training"):
        return "training"
    if kind.startswith("sampling"):
        return "sampling"
    if kind.startswith("adapter"):
        return "adapter"
    if kind.startswith("checkpoint"):
        return "checkpoint"
    return "log"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())
