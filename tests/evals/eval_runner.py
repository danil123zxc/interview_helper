"""Core eval runner helpers."""

import asyncio
import logging
from typing import Any, Dict, List, Tuple

from src.workflow import Workflow
from tests.evals.eval_grading import conciseness_grader_async, hallucination_grader_async
from tests.evals.eval_state import (
    build_context,
    build_subagent_items,
    extract_state_obj,
    extract_subagent_history,
    extract_user_input,
    md_files_to_map,
)

logger = logging.getLogger(__name__)


def _log_state_history_error(exc: Exception) -> None:
    """Log a warning when state history cannot be loaded.

    Args:
        exc: Exception raised while loading state history.

    Returns:
        None.

    Example:
        ```python
        try:
            raise RuntimeError("boom")
        except Exception as exc:
            _log_state_history_error(exc)
        # warning log emitted
        ```
    """
    logger.warning("Failed to load state history: %s", exc)


async def run_agent_for_row(
    row: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], Any, Any]:
    """Run the agent for a dataset row and return md files, final state, and history.

    Args:
        row: Dataset row dict containing user_input and optional context fields.
        defaults: Default values used to build ContextSchema.

    Returns:
        Tuple of (md_files, final_state, history), where md_files is a list of
        {"name": ..., "text": ...} dicts.

    Example:
        ```python
        row = {"user_input": "Help me prepare", "context": {"role": "Engineer"}}
        defaults = {"role": None, "resume": None, "experience_level": "intern"}
        md_files, final_state, history = await run_agent_for_row(row, defaults)
        # md_files is a list of markdown file dicts
        ```
    """
    user_input = extract_user_input(row)
    if not user_input:
        raise SystemExit("Missing user_input in dataset row.")
    context = build_context(row, defaults)
    workflow = Workflow()
    await asyncio.to_thread(workflow.invoke, user_input, context=context)
    final_state = workflow.get_final_state()
    md_files = workflow.list_md_files(workflow.config)
    history = None
    try:
        history = workflow.agent.get_state_history(workflow.config)
    except Exception as exc:
        _log_state_history_error(exc)
    return md_files, final_state, history


async def evaluate_subagent(item: Dict[str, str], example_id: str) -> Dict[str, Any]:
    """Evaluate a single subagent output using conciseness and hallucination.

    Args:
        item: Dict with "subagent_name", "input", and "output" fields.
        example_id: Dataset example identifier.

    Returns:
        Dict with subagent name, example id, and score/comment fields.

    Example:
        ```python
        item = {"subagent_name": "analyze_agent", "input": "history", "output": "text"}
        result = await evaluate_subagent(item, "0")
        # result["subagent_name"] == "analyze_agent"
        # result["conciseness_score"] is a float or None
        ```
    """
    conc_task = conciseness_grader_async(item["input"], item["output"])
    hall_task = hallucination_grader_async(item["input"], item["output"])
    conc, hall = await asyncio.gather(conc_task, hall_task)
    return {
        "example_id": example_id,
        "subagent_name": item["subagent_name"],
        "conciseness_score": conc.get("score"),
        "conciseness_comment": conc.get("comment"),
        "hallucination_score": hall.get("score"),
        "hallucination_comment": hall.get("comment"),
    }


async def evaluate_example(
    row: Dict[str, Any],
    defaults: Dict[str, Any],
    example_id: str,
) -> List[Dict[str, Any]]:
    """Run the agent for one example and evaluate all subagents in parallel.

    Args:
        row: Dataset row dict containing user_input and optional context fields.
        defaults: Default values used to build ContextSchema.
        example_id: Dataset example identifier.

    Returns:
        List of per-subagent evaluation result dicts.

    Example:
        ```python
        row = {"user_input": "Help me prepare", "context": {"role": "Engineer"}}
        results = await evaluate_example(row, {"experience_level": "intern"}, "0")
        # results is a list of dicts with score/comment fields
        ```
    """
    md_files, final_state, history = await run_agent_for_row(row, defaults)
    md_map = md_files_to_map(md_files)
    state_obj = extract_state_obj(final_state)
    history_by_subagent = extract_subagent_history(state_obj, history)
    if not history_by_subagent:
        logger.warning("No subagent history found in final state; using empty inputs.")
    items = build_subagent_items(md_map, history_by_subagent)
    tasks = [evaluate_subagent(item, example_id) for item in items]
    return list(await asyncio.gather(*tasks))
