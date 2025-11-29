import logging
import os


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
