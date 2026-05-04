import io
import logging

from llm import observability


def _capture(level: str = "INFO") -> tuple[io.StringIO, logging.Handler]:
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(observability._ConsoleFormatter(use_color=False))
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    for existing in previous_handlers:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    def restore() -> None:
        root.removeHandler(handler)
        for existing in previous_handlers:
            root.addHandler(existing)
        root.setLevel(previous_level)

    handler.restore = restore  # type: ignore[attr-defined]
    return buffer, handler


def test_log_query_renders_indented_block_with_phase_tag():
    buffer, handler = _capture()
    try:
        observability.log_query("sampling", "select main.fruit", "<request>\n<select/>\n</request>")
    finally:
        handler.restore()  # type: ignore[attr-defined]

    output = buffer.getvalue()
    assert "sampling" in output
    assert "▶ sampling · select main.fruit" in output
    assert "  <request>" in output
    assert "  <select/>" in output


def test_log_tokens_includes_token_count_and_text():
    buffer, handler = _capture()
    try:
        observability.log_tokens("sampling_response", "fruit", "<row>1</row>", [11, 22, 33])
    finally:
        handler.restore()  # type: ignore[attr-defined]

    output = buffer.getvalue()
    assert "◀ sampling_response · fruit · 3 tokens" in output
    assert "  <row>1</row>" in output


def test_emit_progress_renders_step_and_percent_in_message():
    buffer, handler = _capture()
    try:
        observability.emit_progress(
            "training", "step 12/400 loss=0.5", percent=51.2, step=12, total=400
        )
    finally:
        handler.restore()  # type: ignore[attr-defined]

    output = buffer.getvalue().strip()
    assert "training" in output
    assert "step 12/400 loss=0.5" in output
    assert "[12/400]" in output
    assert "(51.2%)" in output


def test_log_section_emits_banner_with_separator():
    buffer, handler = _capture()
    try:
        observability.log_section("boot", "ready · checkpoint=abc", char="═")
    finally:
        handler.restore()  # type: ignore[attr-defined]

    output = buffer.getvalue().strip()
    assert "ready · checkpoint=abc" in output
    assert "═" in output
