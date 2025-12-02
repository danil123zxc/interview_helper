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

    # Avoid re-initializing across Streamlit reruns or multiple imports.
    if sentry_sdk.Hub.current.client is not None:
        return

    traces_sample_rate = float(os.getenv("SENTRY_TRACES", "0.0"))
    profiles_sample_rate = float(os.getenv("SENTRY_PROFILES", "0.0"))
    environment = os.getenv("SENTRY_ENV", "dev")
    release = os.getenv("GIT_COMMIT_SHA")

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        send_default_pii=True,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        integrations=[
            LoggingIntegration(
                level=logging.INFO,        # breadcrumb level
                event_level=logging.ERROR, # send events at ERROR+
            )
        ],
    )
