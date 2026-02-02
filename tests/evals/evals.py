"""
Evaluation runner for subagent outputs.

Workflow:
1) Run the agent on each dataset example.
2) Wait until the run completes.
3) Read the final state markdown files.
4) Evaluate each subagent independently using its system prompt as input
   and its markdown files as output.

Dataset sources:
- --dataset: LangSmith dataset name or UUID (requires LangSmith API key)
- --history: local JSON list of examples
- --inputs-dataset + --context-dataset: separate LangSmith datasets for user inputs and context
- --inputs-history + --context-history: separate local JSON files for user inputs and context

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
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langsmith import Client
from pydantic import BaseModel

from src.schemas import ContextSchema
from src.workflow import Workflow
from src.prompts import (
    analyze_agent_prompt,
    job_posting_ingestor_prompt,
    planner_agent_prompt,
    question_writer_prompt,
    research_agent_prompt,
    synthesis_agent_prompt,
)
from tests.evals.models import ConcisenessEval, HallucinationEval
from tests.evals.prompts import conciseness_prompt, hallucination_prompt

logger = logging.getLogger(__name__)

_JUDGE: Optional[BaseChatModel] = None

SUBAGENT_PROMPTS: Dict[str, str] = {
    "job_posting_ingestor": job_posting_ingestor_prompt,
    "analyze_agent": analyze_agent_prompt,
    "research_agent": research_agent_prompt,
    "question_writer": question_writer_prompt,
    "planner_agent": planner_agent_prompt,
    "synthesis_agent": synthesis_agent_prompt,
}

SUBAGENT_FILES: Dict[str, List[str]] = {
    "job_posting_ingestor": ["job_posting.md"],
    "analyze_agent": ["analysis.md"],
    "research_agent": ["research.md"],
    "question_writer": ["questions.md"],
    "planner_agent": ["prep_plan.md"],
    "synthesis_agent": ["final_response.md"],
}

USER_INPUT_KEYS = ("user_input", "input", "question", "prompt", "user_prompt")
CONTEXT_KEYS = ("role", "resume", "experience_level", "years_of_experience")

METRICS = (
    {
        "key": "conciseness",
        "label": "Conciseness",
        "score_field": "conciseness_score",
        "comment_field": "conciseness_comment",
        "direction": "higher_better",
    },
    {
        "key": "hallucination_rate",
        "label": "Hallucination rate",
        "score_field": "hallucination_score",
        "comment_field": "hallucination_comment",
        "direction": "lower_better",
    },
)


def _get_judge() -> BaseChatModel:
    """Lazily initialize and return the judge chat model."""
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = init_chat_model(model="gpt-5-mini", model_provider="openai", temperature=0)
    return _JUDGE


def _extract_text(resp: Any) -> str:
    """Best-effort stringify of LLM responses."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c)
    return str(content)


async def _call_judge_async(
    prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    judge: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """Call judge LLM and parse structured or JSON-ish output."""
    judge = judge or _get_judge()
    if response_model is not None:
        try:
            structured = judge.with_structured_output(response_model)
            parsed = await structured.ainvoke(prompt)
            if isinstance(parsed, BaseModel):
                return parsed.model_dump()
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            logger.debug("Structured output failed: %s", exc)

    raw = await judge.ainvoke(prompt)
    text = _extract_text(raw).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        parsed = {}

    lower = text.lower()
    if "score" in lower:
        for token in text.replace(",", " ").split():
            try:
                val = float(token)
                parsed.setdefault("score", val)
                break
            except Exception:
                continue
    parsed.setdefault("comment", text or "No comment")
    return parsed


async def _conciseness_grader_async(inp: str, pred: str) -> Dict[str, Any]:
    """Prompt the judge to score conciseness."""
    prompt = conciseness_prompt.format(inputs=inp, outputs=pred)
    result = await _call_judge_async(prompt, response_model=ConcisenessEval)
    comment = result.get("comment", "")
    score_raw = result.get("conciseness")
    if score_raw is None:
        score_raw = result.get("score")
    score: Optional[float]
    try:
        score = float(score_raw) if score_raw is not None else None
    except Exception:
        score = None
    return {"key": "conciseness", "score": score, "comment": comment}


async def _hallucination_grader_async(inp: str, pred: str) -> Dict[str, Any]:
    """Prompt the judge to score hallucination rate."""
    prompt = hallucination_prompt.format(inputs=inp, outputs=pred)
    result = await _call_judge_async(prompt, response_model=HallucinationEval)
    comment = result.get("comment", "")
    halluc = result.get("hallucination")
    if isinstance(halluc, str):
        halluc = halluc.strip().lower() in {"true", "yes", "1"}
    if halluc is None:
        score_raw = result.get("score")
        score = float(score_raw) if score_raw is not None else None
    else:
        score = 1.0 if halluc else 0.0
    return {"key": "hallucination_rate", "score": score, "comment": comment}


def _extract_user_input(row: Dict[str, Any]) -> Optional[str]:
    """Extract the user input string from a dataset row."""
    for key in USER_INPUT_KEYS:
        val = row.get(key)
        if val:
            return str(val)
    return None


def _build_context(
    row: Dict[str, Any],
    defaults: Dict[str, Any],
) -> ContextSchema:
    """Build ContextSchema from a dataset row with fallbacks."""
    context = row.get("context") if isinstance(row.get("context"), dict) else {}
    role = context.get("role") or row.get("role") or defaults.get("role")
    resume = context.get("resume") or row.get("resume") or defaults.get("resume")
    experience_level = (
        context.get("experience_level")
        or row.get("experience_level")
        or defaults.get("experience_level")
        or "intern"
    )
    years_of_experience = (
        context.get("years_of_experience")
        or row.get("years_of_experience")
        or defaults.get("years_of_experience")
    )
    if not role:
        logger.warning("Missing role in dataset row; using 'unknown'.")
        role = "unknown"
    return ContextSchema(
        role=role,
        resume=resume,
        experience_level=experience_level,
        years_of_experience=years_of_experience,
    )


def _has_openai_key() -> bool:
    """Return True if OPENAI_API_KEY is set."""
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_langsmith_key() -> bool:
    """Return True if LangSmith API key env var is set."""
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def _load_history_rows(path: str) -> List[Dict[str, Any]]:
    """Load local JSON rows from a history file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_langsmith_rows(dataset_name: str) -> List[Dict[str, Any]]:
    """Load input rows from a LangSmith dataset."""
    client = Client()
    rows: List[Dict[str, Any]] = []
    for ex in client.list_examples(dataset_name=dataset_name):
        row = dict(ex.inputs or {})
        row["_example_id"] = str(ex.id)
        rows.append(row)
    return rows


def _merge_input_context_rows(
    inputs_rows: List[Dict[str, Any]],
    context_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge input rows with context rows by index."""
    if len(inputs_rows) != len(context_rows):
        raise SystemExit(
            "Inputs and context datasets must have the same length. "
            f"Got {len(inputs_rows)} inputs vs {len(context_rows)} context rows."
        )

    merged: List[Dict[str, Any]] = []
    for input_row, context_row in zip(inputs_rows, context_rows):
        row = dict(input_row)

        context_payload: Dict[str, Any] = {}
        if isinstance(context_row.get("context"), dict):
            context_payload.update(context_row["context"])
        for key in CONTEXT_KEYS:
            if key in context_row and context_row[key] is not None:
                context_payload.setdefault(key, context_row[key])

        if context_payload:
            row["context"] = context_payload

        for key, value in context_row.items():
            row.setdefault(key, value)

        merged.append(row)
    return merged


def _md_files_to_map(md_files: List[Dict[str, str]]) -> Dict[str, str]:
    """Convert list of md file dicts into a name->text mapping."""
    md_map: Dict[str, str] = {}
    for item in md_files:
        name = item.get("name")
        text = item.get("text", "")
        if name:
            md_map[str(name)] = str(text)
    return md_map


def _build_subagent_items(md_map: Dict[str, str]) -> List[Dict[str, str]]:
    """Build per-subagent input/output pairs for evaluation."""
    items: List[Dict[str, str]] = []
    for subagent_name, prompt in SUBAGENT_PROMPTS.items():
        file_names = SUBAGENT_FILES.get(subagent_name, [])
        outputs: List[str] = []
        for file_name in file_names:
            text = md_map.get(file_name)
            if text:
                outputs.append(f"# {file_name}\\n{text}".strip())
        if not outputs and file_names:
            outputs.append("missing: " + ", ".join(file_names))
        output_text = "\\n\\n".join(outputs).strip()
        items.append(
            {
                "subagent_name": subagent_name,
                "input": prompt,
                "output": output_text,
            }
        )
    return items


async def _run_agent_for_row(row: Dict[str, Any], defaults: Dict[str, Any]) -> List[Dict[str, str]]:
    """Run the agent for a dataset row and return md file dicts."""
    user_input = _extract_user_input(row)
    if not user_input:
        raise SystemExit("Missing user_input in dataset row.")
    context = _build_context(row, defaults)
    workflow = Workflow()
    await asyncio.to_thread(workflow.invoke, user_input, context=context, config=workflow.config)
    return workflow.list_md_files(workflow.config)


async def _evaluate_subagent(item: Dict[str, str], example_id: str) -> Dict[str, Any]:
    """Evaluate a single subagent output using conciseness and hallucination."""
    conc_task = _conciseness_grader_async(item["input"], item["output"])
    hall_task = _hallucination_grader_async(item["input"], item["output"])
    conc, hall = await asyncio.gather(conc_task, hall_task)
    return {
        "example_id": example_id,
        "subagent_name": item["subagent_name"],
        "conciseness_score": conc.get("score"),
        "conciseness_comment": conc.get("comment"),
        "hallucination_score": hall.get("score"),
        "hallucination_comment": hall.get("comment"),
    }


async def _evaluate_example(
    row: Dict[str, Any],
    defaults: Dict[str, Any],
    example_id: str,
) -> List[Dict[str, Any]]:
    """Run agent for one example and evaluate all subagents in parallel."""
    md_files = await _run_agent_for_row(row, defaults)
    md_map = _md_files_to_map(md_files)
    items = _build_subagent_items(md_map)
    tasks = [
        _evaluate_subagent(item, example_id)
        for item in items
    ]
    return list(await asyncio.gather(*tasks))


def _summarize_by_subagent(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute per-subagent summary stats for each metric."""
    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        subagent = row.get("subagent_name", "unknown")
        summary.setdefault(subagent, {})
        for metric in METRICS:
            scores = summary[subagent].setdefault(metric["key"], {"scores": []})
            score = row.get(metric["score_field"])
            if isinstance(score, (int, float)):
                scores["scores"].append(float(score))

    for subagent, metrics in summary.items():
        for metric_key, data in metrics.items():
            scores = data.get("scores", [])
            metrics[metric_key] = {
                "count": len(scores),
                "avg": (sum(scores) / len(scores)) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
            }
    return summary


def _normalize_comment(text: Optional[str], limit: int = 160) -> str:
    """Normalize whitespace and truncate comments for tables."""
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _format_score(score: Optional[float]) -> str:
    """Format a numeric score for display."""
    if score is None:
        return "N/A"
    return f"{score:.2f}"


def _render_markdown_summary(
    summary: Dict[str, Dict[str, Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    dataset_label: str,
) -> str:
    """Render a GitHub Actions summary for eval results."""
    lines = ["## Subagent evals", f"Dataset: `{dataset_label}`", ""]
    for subagent, metrics in summary.items():
        lines.append(f"### {subagent}")
        for metric in METRICS:
            stats = metrics.get(metric["key"], {})
            direction = "higher is better" if metric["direction"] == "higher_better" else "lower is better"
            lines.append(f"- {metric['label']} ({direction}): avg={_format_score(stats.get('avg'))}")
        lines.append("")

    lines.append("| Example | Subagent | Conciseness | Hallucination rate | Conciseness comment | Hallucination comment |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        conc_comment = _normalize_comment(row.get("conciseness_comment"))
        hall_comment = _normalize_comment(row.get("hallucination_comment"))
        conc_comment = conc_comment.replace("|", "\\|")
        hall_comment = hall_comment.replace("|", "\\|")
        lines.append(
            "| {example_id} | {subagent} | {conc} | {hall} | {conc_comment} | {hall_comment} |".format(
                example_id=row.get("example_id", "-"),
                subagent=row.get("subagent_name", "unknown"),
                conc=_format_score(row.get("conciseness_score")),
                hall=_format_score(row.get("hallucination_score")),
                conc_comment=conc_comment,
                hall_comment=hall_comment,
            )
        )
    lines.append("")
    return "\\n".join(lines)


def _write_github_summary(markdown: str) -> None:
    """Append markdown to GitHub Actions step summary if available."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(markdown)
        if not markdown.endswith("\\n"):
            f.write("\\n")


async def _run_evals_async(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run agent and evals over the chosen dataset."""
    if not _has_openai_key():
        raise SystemExit("OPENAI_API_KEY is required to run evals.")

    defaults = {
        "role": args.role,
        "resume": args.resume,
        "experience_level": args.experience_level,
        "years_of_experience": args.years_of_experience,
    }

    if args.inputs_dataset or args.context_dataset:
        if not (args.inputs_dataset and args.context_dataset):
            raise SystemExit("--inputs-dataset and --context-dataset must be provided together.")
        if not _has_langsmith_key():
            raise SystemExit("LANGCHAIN_API_KEY or LANGSMITH_API_KEY required for LangSmith datasets.")
        input_rows = _load_langsmith_rows(args.inputs_dataset)
        context_rows = _load_langsmith_rows(args.context_dataset)
        rows = _merge_input_context_rows(input_rows, context_rows)
        dataset_label = f"{args.inputs_dataset} + {args.context_dataset}"
    elif args.inputs_history or args.context_history:
        if not (args.inputs_history and args.context_history):
            raise SystemExit("--inputs-history and --context-history must be provided together.")
        input_rows = _load_history_rows(args.inputs_history)
        context_rows = _load_history_rows(args.context_history)
        rows = _merge_input_context_rows(input_rows, context_rows)
        dataset_label = f"{args.inputs_history} + {args.context_history}"
    elif args.dataset:
        if not _has_langsmith_key():
            raise SystemExit("LANGCHAIN_API_KEY or LANGSMITH_API_KEY required for --dataset.")
        rows = _load_langsmith_rows(args.dataset)
        dataset_label = args.dataset
    else:
        if not args.history:
            raise SystemExit("--history is required when not using --dataset.")
        rows = _load_history_rows(args.history)
        dataset_label = args.history

    all_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        example_id = row.get("_example_id") or str(idx)
        results = await _evaluate_example(row, defaults, example_id)
        all_rows.extend(results)

    summary = _summarize_by_subagent(all_rows)
    markdown = _render_markdown_summary(summary, all_rows, dataset_label)
    _write_github_summary(markdown)

    return all_rows


def main() -> None:
    """CLI entrypoint for running subagent evals."""
    parser = argparse.ArgumentParser(description="Run subagent evals over a dataset.")
    parser.add_argument("--history", help="Path to local JSON dataset.")
    parser.add_argument("--dataset", help="LangSmith dataset name or UUID.")
    parser.add_argument("--inputs-history", help="Path to local JSON dataset with user inputs.")
    parser.add_argument("--context-history", help="Path to local JSON dataset with context.")
    parser.add_argument("--inputs-dataset", help="LangSmith dataset name/UUID with user inputs.")
    parser.add_argument("--context-dataset", help="LangSmith dataset name/UUID with context.")
    parser.add_argument("--role", help="Default role if not provided by examples.")
    parser.add_argument("--resume", help="Default resume if not provided by examples.")
    parser.add_argument("--experience-level", default="intern", help="Default experience level.")
    parser.add_argument("--years-of-experience", type=int, help="Default years of experience.")
    args = parser.parse_args()

    if not args.dataset and not args.history:
        default_history = Path("data/subagent_outputs.sample.json")
        if default_history.exists():
            args.history = str(default_history)
            logger.info("Using default history dataset: %s", args.history)

    results = asyncio.run(_run_evals_async(args))
    print(f"Completed subagent evals: {len(results)} rows")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
