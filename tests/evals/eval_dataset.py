"""Dataset loading and LangSmith upload helpers."""

import json
import logging
import os
import re
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from langsmith import Client

logger = logging.getLogger(__name__)


def has_openai_key() -> bool:
    """Check whether the OpenAI API key is available.

    Args:
        None.

    Returns:
        True if OPENAI_API_KEY is set, otherwise False.

    Example:
        ```python
        os.environ["OPENAI_API_KEY"] = "test-key"
        has_openai_key()
        # True
        ```
    """
    return bool(os.getenv("OPENAI_API_KEY"))


def has_langsmith_key() -> bool:
    """Check whether a LangSmith API key is available.

    Args:
        None.

    Returns:
        True if LANGSMITH_API_KEY or LANGCHAIN_API_KEY is set, otherwise False.

    Example:
        ```python
        os.environ["LANGSMITH_API_KEY"] = "test-key"
        has_langsmith_key()
        # True
        ```
    """
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def sanitize_dataset_name(name: str) -> str:
    """Normalize a dataset name to safe characters.

    Args:
        name: Input dataset name string.

    Returns:
        Sanitized name using only letters, digits, underscores, and dashes.

    Example:
        ```python
        sanitize_dataset_name("My Dataset 2026/02")
        # "My-Dataset-2026-02"
        ```
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    return cleaned or "evals-dataset"


def default_dataset_name(path: str, suffix: Optional[str] = None) -> str:
    """Build a default dataset name from a file path.

    Args:
        path: File path used to derive the base name.
        suffix: Optional suffix appended to the dataset name.

    Returns:
        Normalized dataset name string.

    Example:
        ```python
        default_dataset_name("tests/evals/dataset.py", "module")
        # "evals-dataset-module"
        ```
    """
    base = sanitize_dataset_name(Path(path).stem)
    name = f"evals-{base}"
    if suffix:
        name = f"{name}-{suffix}"
    return name


def get_or_create_dataset(client: Client, dataset_name: str):
    """Fetch a LangSmith dataset or create it if missing.

    Args:
        client: LangSmith Client instance.
        dataset_name: Dataset name to read or create.

    Returns:
        LangSmith dataset object.

    Example:
        ```python
        client = Client()
        ds = get_or_create_dataset(client, "evals-dataset")
        # ds.name == "evals-dataset"
        ```
    """
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        logger.info("Creating LangSmith dataset: %s", dataset_name)
        return client.create_dataset(dataset_name=dataset_name)


def dataset_has_examples(client: Client, dataset_id: str) -> bool:
    """Check whether a LangSmith dataset already has examples.

    Args:
        client: LangSmith Client instance.
        dataset_id: Dataset id to inspect.

    Returns:
        True if at least one example exists, otherwise False.

    Example:
        ```python
        client = Client()
        has_any = dataset_has_examples(client, "dataset-id")
        # has_any is True or False
        ```
    """
    try:
        return next(client.list_examples(dataset_id=dataset_id, limit=1), None) is not None
    except Exception:
        return False


def upload_rows_to_langsmith(dataset_name: str, rows: List[Dict[str, Any]]) -> bool:
    """Upload local rows to LangSmith as input-only examples.

    Args:
        dataset_name: LangSmith dataset name to create or update.
        rows: List of input dicts to upload.

    Returns:
        True if examples were uploaded, False if skipped (already populated).

    Example:
        ```python
        rows = [{"user_input": "Help me prep"}]
        uploaded = upload_rows_to_langsmith("evals-dataset", rows)
        # uploaded is True or False
        ```
    """
    client = Client()
    dataset = get_or_create_dataset(client, dataset_name)
    if dataset_has_examples(client, dataset.id):
        logger.info("LangSmith dataset %s already has examples; skipping upload.", dataset_name)
        return False
    examples = [{"inputs": row} for row in rows]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    logger.info("Uploaded %d examples to LangSmith dataset: %s", len(rows), dataset_name)
    return True


def should_upload() -> bool:
    """Check whether local datasets should be uploaded to LangSmith.

    Args:
        None.

    Returns:
        True if a LangSmith API key is present, otherwise False.

    Example:
        ```python
        os.environ["LANGCHAIN_API_KEY"] = "test-key"
        should_upload()
        # True
        ```
    """
    return has_langsmith_key()


def load_history_rows(path: str) -> List[Dict[str, Any]]:
    """Load local JSON rows from a file.

    Args:
        path: Path to a JSON file containing a list of dicts.

    Returns:
        List of row dictionaries from the JSON file.

    Example:
        ```python
        rows = load_history_rows("data/inputs.sample.json")
        # rows[0]["user_input"] is a string
        ```
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_module_rows(path: str) -> List[Dict[str, Any]]:
    """Load examples from a Python module that defines EXAMPLES.

    Args:
        path: Path to a Python file containing EXAMPLES (list of dicts).

    Returns:
        List of input row dictionaries.

    Example:
        ```python
        # dataset.py contains: EXAMPLES = [{"inputs": {"user_input": "Hi"}}]
        rows = load_module_rows("tests/evals/dataset.py")
        # rows == [{"user_input": "Hi"}]
        ```
    """
    module_path = Path(path)
    if not module_path.exists():
        raise SystemExit(f"Dataset module not found: {path}")

    spec = importlib.util.spec_from_file_location("evals_dataset_module", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to import dataset module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    examples = getattr(module, "EXAMPLES", [])
    if not isinstance(examples, list) or not examples:
        raise SystemExit("EXAMPLES is missing or empty in dataset module.")

    rows: List[Dict[str, Any]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        inputs = ex.get("inputs") if "inputs" in ex else ex
        if isinstance(inputs, dict):
            rows.append(inputs)

    if not rows:
        raise SystemExit("No valid inputs found in dataset module EXAMPLES.")
    return rows


def load_langsmith_rows(dataset_name: str) -> List[Dict[str, Any]]:
    """Load input rows from a LangSmith dataset.

    Args:
        dataset_name: LangSmith dataset name or UUID.

    Returns:
        List of row dicts with an added "_example_id" field.

    Example:
        ```python
        rows = load_langsmith_rows("evals-dataset")
        # rows[0]["_example_id"] is set when examples exist
        ```
    """
    client = Client()
    rows: List[Dict[str, Any]] = []
    for ex in client.list_examples(dataset_name=dataset_name):
        row = dict(ex.inputs or {})
        row["_example_id"] = str(ex.id)
        rows.append(row)
    return rows
