"""Evaluation runner for subagent outputs.

Workflow:
1) Run the agent on each dataset example.
2) Wait until the run completes.
3) Read the final state markdown files.
4) Evaluate each subagent independently using its execution history from
   the final state as input and its markdown files as output.

Dataset sources:
- --dataset: LangSmith dataset name/UUID or a local path (.json or .py).
If not provided, uses tests/evals/dataset/dataset.py (EXAMPLES).
Local datasets are uploaded to LangSmith automatically when API keys are available.

Local JSON format (minimal):
[
  {
    "user_input": "Help me prepare for ...",
    "context": {
      "role": "AI engineer",
      "resume": "...",
      "experience_level": "intern",
      "years_of_experience": 0
    }
  }
]
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.logging_config import setup_logging
from tests.evals.dataset.eval_dataset import (
    default_dataset_name,
    has_langsmith_key,
    has_openai_key,
    load_history_rows,
    load_langsmith_rows,
    load_module_rows,
    should_upload,
    upload_rows_to_langsmith,
)
from tests.evals.eval_reporting import (
    render_markdown_summary,
    summarize_by_subagent,
    write_github_summary,
)
from tests.evals.eval_runner import evaluate_example

logger = logging.getLogger(__name__)


async def _run_evals_async(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run agent and evals over the chosen dataset.

    Args:
        args: Parsed CLI arguments with an optional dataset path or LangSmith name.

    Returns:
        Flattened list of evaluation result dicts (one per subagent per example).

    Example:
        ```python
        args = argparse.Namespace(dataset="tests/evals/dataset/dataset.py")
        results = asyncio.run(_run_evals_async(args))
        # len(results) >= 1 when dataset has examples
        ```
    """
    if not has_openai_key():
        raise SystemExit("OPENAI_API_KEY is required to run evals.")

    defaults = {
        "role": None,
        "resume": None,
        "experience_level": "intern",
        "years_of_experience": None,
    }

    uploaded_names: List[str] = []

    if args.dataset:
        dataset_path = Path(args.dataset)
        if dataset_path.exists():
            suffix = dataset_path.suffix.lower()
            if suffix == ".py":
                rows = load_module_rows(str(dataset_path))
                dataset_label = str(dataset_path)
                if should_upload():
                    dataset_name = default_dataset_name(str(dataset_path), "module")
                    try:
                        if upload_rows_to_langsmith(dataset_name, rows):
                            uploaded_names.append(dataset_name)
                    except Exception as exc:
                        logger.warning("LangSmith upload failed: %s", exc)
            elif suffix == ".json":
                rows = load_history_rows(str(dataset_path))
                dataset_label = str(dataset_path)
                if should_upload():
                    dataset_name = default_dataset_name(str(dataset_path), "json")
                    try:
                        if upload_rows_to_langsmith(dataset_name, rows):
                            uploaded_names.append(dataset_name)
                    except Exception as exc:
                        logger.warning("LangSmith upload failed: %s", exc)
            else:
                raise SystemExit("Unsupported dataset path. Use a .json or .py file.")
        else:
            if not has_langsmith_key():
                raise SystemExit("LANGCHAIN_API_KEY or LANGSMITH_API_KEY required for LangSmith datasets.")
            rows = load_langsmith_rows(args.dataset)
            dataset_label = args.dataset
    else:
        module_path = Path("tests/evals/dataset/dataset.py")
        if not module_path.exists():
            raise SystemExit("Default dataset module not found: tests/evals/dataset/dataset.py")
        rows = load_module_rows(str(module_path))
        dataset_label = str(module_path)
        if should_upload():
            dataset_name = default_dataset_name(str(module_path), "module")
            try:
                if upload_rows_to_langsmith(dataset_name, rows):
                    uploaded_names.append(dataset_name)
            except Exception as exc:
                logger.warning("LangSmith upload failed: %s", exc)

    if uploaded_names:
        dataset_label = f"{dataset_label} (uploaded to {', '.join(uploaded_names)})"

    all_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        example_id = row.get("_example_id") or str(idx)
        results = await evaluate_example(row, defaults, example_id)
        all_rows.extend(results)

    summary = summarize_by_subagent(all_rows)
    markdown = render_markdown_summary(summary, all_rows, dataset_label)
    write_github_summary(markdown)

    return all_rows


def main() -> None:
    """CLI entrypoint for running subagent evals.

    Args:
        None. Arguments are parsed from sys.argv.

    Returns:
        None. Prints a completion line to stdout.

    Example:
        ```python
        # CLI usage:
        # python tests/evals/evals.py --dataset tests/evals/dataset/dataset.py
        #
        # Output includes:
        # "Completed subagent evals: <N> rows"
        ```
    """
    parser = argparse.ArgumentParser(description="Run subagent evals over a dataset.")
    parser.add_argument(
        "--dataset",
        help="LangSmith dataset name/UUID or local path (.json or .py).",
    )
    args = parser.parse_args()

    results = asyncio.run(_run_evals_async(args))
    print(f"Completed subagent evals: {len(results)} rows")


if __name__ == "__main__":
    setup_logging()
    main()
