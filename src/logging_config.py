import contextvars
import logging
import os
from contextlib import contextmanager

import sentry_sdk

_thread_id_var = contextvars.ContextVar("thread_id", default="-")
_run_id_var = contextvars.ContextVar("run_id", default="-")


class ContextFilter(logging.Filter):
    """Inject thread/run context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "thread_id"):
            record.thread_id = _thread_id_var.get() or "-"
        if not hasattr(record, "run_id"):
            record.run_id = _run_id_var.get() or "-"
        return True


@contextmanager
def logging_context(*, thread_id: str | None = None, run_id: str | None = None):
    """Temporarily set thread/run context for logs."""
    tokens = []
    if thread_id is not None:
        tokens.append((_thread_id_var, _thread_id_var.set(thread_id)))
    if run_id is not None:
        tokens.append((_run_id_var, _run_id_var.set(run_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _attach_context_filter() -> None:
    root_logger = logging.getLogger()
    context_filter = ContextFilter()
    for handler in root_logger.handlers:
        if not any(isinstance(f, ContextFilter) for f in handler.filters):
            handler.addFilter(context_filter)


def _set_noisy_loggers(level: str) -> None:
    for name in (
        "urllib3",
        "httpx",
        "httpcore",
        "openai",
        "langchain",
        "langgraph",
        "langsmith",
        "tavily",
        "sentry_sdk",
    ):
        logging.getLogger(name).setLevel(level)

def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Level can be overridden via LOG_LEVEL env."""
    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    if logging.getLogger().handlers:
        # Already configured; just update level.
        logging.getLogger().setLevel(log_level)
        _attach_context_filter()
        _set_noisy_loggers(os.getenv("NOISY_LOG_LEVEL", "WARNING").upper())
        return

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] [thread=%(thread_id)s run=%(run_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _attach_context_filter()
    _set_noisy_loggers(os.getenv("NOISY_LOG_LEVEL", "WARNING").upper())

def init_sentry() -> None:
    """Initialize Sentry SDK for error tracking (idempotent)."""
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return
    
    sentry_sdk.init(
        dsn=dsn,
        # Add data like request headers and IP for users,
        # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
        send_default_pii=True,
        # Enable sending logs to Sentry
        enable_logs=True,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for tracing.
        traces_sample_rate=1.0,
    )
