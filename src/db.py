from contextlib import contextmanager
import logging
import os
from urllib.parse import urlparse

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from src.workflow import Workflow

logger = logging.getLogger(__name__)

@contextmanager
def workflow_ctx():
    """Provide a Workflow with Postgres-backed store/checkpointer.

    Args:
        None. Uses the DB_URL environment variable.

    Returns:
        Context manager yielding a configured Workflow instance.

    Example:
        ```python
        import os
        from src.db import workflow_ctx

        os.environ["DB_URL"] = "postgresql://user:pass@localhost:5432/app"
        with workflow_ctx() as wf:
            wf.invoke("Hello", context=some_context)
        ```
    """

    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is required to use Postgres-backed workflows.")

    parsed = urlparse(db_url)
    if not parsed.scheme or not parsed.hostname:
        raise RuntimeError(
            "DB_URL must be a valid Postgres connection string, "
            "e.g., postgresql://user:pass@host:5432/dbname"
        )
    if parsed.scheme not in {"postgres", "postgresql"}:
        logger.warning("DB_URL scheme is %s; expected postgres/postgresql.", parsed.scheme)

    with (
        PostgresStore.from_conn_string(db_url) as store,
        PostgresSaver.from_conn_string(db_url) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        yield Workflow(store=store, checkpointer=checkpointer)

