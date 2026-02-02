"""
Evaluation runner for subagent outputs.

This script follows the LangSmith evaluate-LLM-application structure:
1) Define an application (target function).
2) Select a dataset (LangSmith dataset or local JSON history).
3) Define evaluators.
4) Run evaluation.

Local JSON formats supported (passed via --history):

Format A (subagent markdown history):
[
  {
    "subagent_name": "analyze_agent",
    "system_prompt": "... full system prompt passed to the subagent ...",
    "user_messages": ["human prompt 1", "human prompt 2"],
    "md_files": [
      {"name": "analysis.md", "text": "...markdown content..."}
    ]
  }
]

Format B (flat dataset):
[
  {
    "subagent_name": "analyze_agent",
    "system_prompt": "You task is to analyze a resume and a job posting.",
    "user_prompt": "Analyze this resume vs the attached job posting and give gaps.",
    "subagent_output": "Fit summary: ..."
  }
]

The evals grade:
- conciseness (0-1, higher is better)
- hallucination rate (0-1, lower is better)
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langsmith import Client, traceable
from evals.prompts import conciseness_prompt, hallucination_prompt

logger = logging.getLogger(__name__)

_JUDGE: Optional[BaseChatModel] = None


# Step 1. Define your application (target function)
@traceable
def subagent_output_app(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stored subagent output so we can judge it."""
    return {"output": inputs.get("output", "")}


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


def _call_judge(prompt: str, judge: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    """Call judge LLM and normalize JSON-ish score/comment output."""
    judge = judge or _get_judge()
    raw = judge.invoke(prompt)
    text = _extract_text(raw).strip()

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}
        lower = text.lower()
        if "score" in lower:
            for token in text.replace(",", " ").split():
                try:
                    val = float(token)
                    if 0 <= val <= 1:
                        parsed["score"] = val
                        break
                except Exception:
                    continue
        parsed.setdefault("comment", text or "No comment")

    score = parsed.get("score")
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None
    return {"score": score, "comment": parsed.get("comment", text)}


def _conciseness_grader(inp: str, pred: str) -> Dict[str, Any]:
    """Prompt the judge to score conciseness."""
    prompt = f"""
You score subagent outputs for conciseness.
Context (system + human prompts):
{inp}

Subagent output:
{pred}

Return JSON: {{"score": number between 0 and 1, "comment": "brief reason + trim suggestion"}}.
Score 1 = fully concise and on-topic. Score 0 = verbose, repetitive, off-task.
"""
    result = _call_judge(prompt)
    result["key"] = "conciseness"
    return result


def _hallucination_grader(inp: str, pred: str) -> Dict[str, Any]:
    """Prompt the judge to score hallucination rate."""
    prompt = f"""
You score subagent outputs for hallucination rate.
Use ONLY the provided context. A hallucination is any factual claim in the output
that is not supported by the context.

Context (system + human prompts + optional extra context):
{inp}

Subagent output:
{pred}

Return JSON: {{"score": number between 0 and 1, "comment": "brief reason + note unsupported claims"}}.
Score 0 = no unsupported claims (or no factual claims). Score 1 = mostly unsupported.
"""
    result = _call_judge(prompt)
    result["key"] = "hallucination_rate"
    return result


# Step 3. Define evaluators

def conciseness_eval(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """LLM-judge conciseness of the output given the context."""
    _ = reference_outputs
    return _conciseness_grader(str(inputs.get("input", "")), str(outputs.get("output", "")))


def hallucination_rate_eval(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """LLM-judge hallucination rate against provided context."""
    _ = reference_outputs
    return _hallucination_grader(str(inputs.get("input", "")), str(outputs.get("output", "")))


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


def _coerce_list(val: Any) -> List[str]:
    """Normalize a scalar or list value into a list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    return [str(val)]


def _load_history(path: str) -> List[Dict[str, Any]]:
    """Load a local JSON history file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_examples(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert local history records into LangSmith example dicts."""
    examples: List[Dict[str, Any]] = []
    for row in history:
        user_messages = _coerce_list(
            row.get("user_messages") or row.get("user_prompt") or row.get("user_message")
        )
        md_files = row.get("md_files") or []
        md_chunks = []
        for f in md_files:
            name = f.get("name", "file")
            text = f.get("text", "")
            md_chunks.append(f"# {name}\n{text}".strip())

        output = "\n\n".join(md_chunks).strip()
        if not output:
            output = str(row.get("subagent_output") or row.get("output") or "").strip()

        ctx_parts = [
            f"System prompt:\n{str(row.get('system_prompt', '')).strip()}",
            "User messages:\n" + "\n".join(user_messages),
        ]
        extra_context = str(row.get("context7") or row.get("context") or "").strip()
        if extra_context:
            ctx_parts.append("Extra context:\n" + extra_context)

        inputs = {
            "input": "\n\n".join(ctx_parts).strip(),
            "output": output,
            "subagent_name": row.get("subagent_name", "unknown"),
        }
        examples.append({"inputs": inputs, "outputs": {}})
    return examples


def _has_openai_key() -> bool:
    """Return True if OPENAI_API_KEY is set."""
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_langsmith_key() -> bool:
    """Return True if LangSmith API key env var is set."""
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


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


def _extract_eval_result(
    eval_results: Iterable[Any], key: str
) -> Tuple[Optional[float], Optional[str]]:
    """Extract a specific evaluator result from a list of results."""
    for result in eval_results:
        if isinstance(result, dict):
            result_key = result.get("key")
            score = result.get("score")
            comment = result.get("comment")
        else:
            result_key = getattr(result, "key", None)
            score = getattr(result, "score", None)
            comment = getattr(result, "comment", None)
        if result_key == key:
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            return score, comment
    return None, None


def _collect_langsmith_rows(results: Any) -> List[Dict[str, Any]]:
    """Collect per-example evaluator scores from LangSmith results."""
    rows: List[Dict[str, Any]] = []
    for row in results:
        example_inputs = getattr(row.get("example"), "inputs", {}) or {}
        row_data: Dict[str, Any] = {
            "subagent_name": example_inputs.get("subagent_name", "unknown")
        }
        eval_results = (row.get("evaluation_results") or {}).get("results", [])
        for metric in METRICS:
            score, comment = _extract_eval_result(eval_results, metric["key"])
            row_data[metric["score_field"]] = score
            row_data[metric["comment_field"]] = comment
        rows.append(row_data)
    return rows


def _run_local_eval(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run evaluators locally without LangSmith logging."""
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        inputs = ex.get("inputs", {})
        outputs = subagent_output_app(inputs)
        row_data: Dict[str, Any] = {
            "subagent_name": inputs.get("subagent_name", "unknown")
        }
        conc = conciseness_eval(inputs, outputs)
        hall = hallucination_rate_eval(inputs, outputs)
        row_data["conciseness_score"] = conc.get("score")
        row_data["conciseness_comment"] = conc.get("comment")
        row_data["hallucination_score"] = hall.get("score")
        row_data["hallucination_comment"] = hall.get("comment")
        rows.append(row_data)
    return rows


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute summary stats for each metric across rows."""
    summary: Dict[str, Dict[str, Any]] = {}
    for metric in METRICS:
        scores = [
            r.get(metric["score_field"])
            for r in rows
            if isinstance(r.get(metric["score_field"]), (int, float))
        ]
        summary[metric["key"]] = {
            "count": len(rows),
            "scored": len(scores),
            "missing": len(rows) - len(scores),
            "avg": (sum(scores) / len(scores)) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }
    return summary


def _render_markdown_summary(
    summary: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    dataset_label: str,
    experiment_name: Optional[str],
) -> str:
    """Render a GitHub Actions summary table for eval results."""
    lines = ["## Evals", f"Dataset: `{dataset_label}`"]
    if experiment_name:
        lines.append(f"LangSmith experiment: `{experiment_name}`")
    lines.append("")
    for metric in METRICS:
        stats = summary.get(metric["key"], {})
        direction = "higher is better" if metric["direction"] == "higher_better" else "lower is better"
        lines.append(f"### {metric['label']} ({direction})")
        lines.append(
            "Examples: {count} (scored {scored}, missing {missing})".format(**stats)
        )
        if stats.get("avg") is not None:
            lines.append(
                "Average score: {avg:.2f} (min {min:.2f}, max {max:.2f})".format(
                    **stats
                )
            )
        lines.append("")

    lines.append("| Subagent | Conciseness | Hallucination rate | Conciseness comment | Hallucination comment |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        conc_comment = _normalize_comment(row.get("conciseness_comment"))
        hall_comment = _normalize_comment(row.get("hallucination_comment"))
        conc_comment = conc_comment.replace("|", "\\|")
        hall_comment = hall_comment.replace("|", "\\|")
        lines.append(
            "| {subagent} | {conc} | {hall} | {conc_comment} | {hall_comment} |".format(
                subagent=row.get("subagent_name", "unknown"),
                conc=_format_score(row.get("conciseness_score")),
                hall=_format_score(row.get("hallucination_score")),
                conc_comment=conc_comment,
                hall_comment=hall_comment,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_github_summary(markdown: str) -> None:
    """Append markdown to GitHub Actions step summary if available."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(markdown)
        if not markdown.endswith("\n"):
            f.write("\n")


# Step 2. Select dataset (LangSmith dataset or local JSON)
# Step 4. Run evaluation

def main() -> None:
    """CLI entrypoint for running evals locally or via LangSmith."""
    parser = argparse.ArgumentParser(description="Run evals over subagent outputs.")
    parser.add_argument(
        "--history",
        help="Path to local JSON history dataset.",
    )
    parser.add_argument(
        "--dataset",
        help="LangSmith dataset name or UUID (to run evals in LangSmith).",
    )
    parser.add_argument(
        "--sync-dataset",
        action="store_true",
        help="If set, upsert local --history into the LangSmith dataset before evaluating.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="subagent-evals",
        help="Prefix for the LangSmith experiment name.",
    )
    parser.add_argument(
        "--langsmith",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to log results to LangSmith (auto=only if --dataset and key are set).",
    )
    args = parser.parse_args()

    if not _has_openai_key():
        raise SystemExit("OPENAI_API_KEY is required to run evals.")

    use_langsmith: bool
    if args.langsmith == "on":
        if not _has_langsmith_key():
            raise SystemExit("LANGCHAIN_API_KEY or LANGSMITH_API_KEY required for --langsmith on.")
        if not args.dataset:
            raise SystemExit("--dataset is required for --langsmith on.")
        use_langsmith = True
    elif args.langsmith == "auto":
        use_langsmith = bool(args.dataset and _has_langsmith_key())
    else:
        use_langsmith = False

    experiment_name: Optional[str] = None
    dataset_label: str

    if use_langsmith:
        ls_client = Client()
        if args.sync_dataset:
            if not args.history:
                raise SystemExit("--history is required when using --sync-dataset.")
            history = _load_history(args.history)
            if not history:
                raise SystemExit("No history entries found.")
            examples = _build_examples(history)
            if not ls_client.has_dataset(dataset_name=args.dataset):
                ls_client.create_dataset(dataset_name=args.dataset)
            ls_client.create_examples(dataset_name=args.dataset, examples=examples)
        dataset_label = args.dataset
        results = ls_client.evaluate(
            subagent_output_app,
            data=args.dataset,
            evaluators=[conciseness_eval, hallucination_rate_eval],
            experiment_prefix=args.experiment_prefix,
            description="Conciseness and hallucination-rate evals for subagent outputs.",
            max_concurrency=0,
        )
        rows = _collect_langsmith_rows(results)
        experiment_name = getattr(results, "experiment_name", None)
    else:
        if not args.history:
            raise SystemExit("--history is required when not using LangSmith.")
        history = _load_history(args.history)
        if not history:
            raise SystemExit("No history entries found.")
        examples = _build_examples(history)
        dataset_label = args.history
        rows = _run_local_eval(examples)

    summary = _summarize_rows(rows)

    print("Evals")
    print(f"Dataset: {dataset_label}")
    if experiment_name:
        print(f"LangSmith experiment: {experiment_name}")
    for metric in METRICS:
        stats = summary.get(metric["key"], {})
        direction = "higher is better" if metric["direction"] == "higher_better" else "lower is better"
        print(f"{metric['label']} ({direction})")
        print(
            "Examples: {count} (scored {scored}, missing {missing})".format(**stats)
        )
        if stats.get("avg") is not None:
            print(
                "Average score: {avg:.2f} (min {min:.2f}, max {max:.2f})".format(
                    **stats
                )
            )
    for row in rows:
        conc_comment = _normalize_comment(row.get("conciseness_comment"))
        hall_comment = _normalize_comment(row.get("hallucination_comment"))
        print(
            "- {name}: conciseness {conc} | hallucination {hall}"
            " | conc: {conc_comment} | hall: {hall_comment}".format(
                name=row.get("subagent_name", "unknown"),
                conc=_format_score(row.get("conciseness_score")),
                hall=_format_score(row.get("hallucination_score")),
                conc_comment=conc_comment,
                hall_comment=hall_comment,
            )
        )

    markdown = _render_markdown_summary(summary, rows, dataset_label, experiment_name)
    _write_github_summary(markdown)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
