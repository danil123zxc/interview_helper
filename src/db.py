from src.workflow import Workflow
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from contextlib import contextmanager
import os

@contextmanager
def workflow_ctx():
    """Provide a workflow with Postgres-backed"""

    db_url = os.getenv("DB_URL")

    with (
        PostgresStore.from_conn_string(db_url) as store,
        PostgresSaver.from_conn_string(db_url) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        yield Workflow(store=store, checkpointer=checkpointer)

