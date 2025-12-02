import logging
import os
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Level can be overridden via LOG_LEVEL env."""
    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    if logging.getLogger().handlers:
        # Already configured; just update level.
        logging.getLogger().setLevel(log_level)
        return

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
