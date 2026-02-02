"""
Conciseness evaluation for subagent outputs.

Expected input data (JSON file passed via --history) supports two formats:

Format A (subagent markdown history):
[
  {
    "subagent_name": "analyze_agent",
    "system_prompt": "... full system prompt passed to the subagent ...",
    "user_messages": ["human prompt 1", "human prompt 2"],  # or a single string
    "md_files": [
      {"name": "analysis.md", "text": "...markdown content..."},
      {"name": "extra.md", "text": "...optional..."}
    ]
  },
  ...
]

Format B (flat dataset):
[
  {
    "subagent_name": "analyze_agent",
    "system_prompt": "You task is to analyze a resume and a job posting.",
    "user_prompt": "Analyze this resume vs the attached job posting and give gaps.",
    "subagent_output": "Fit summary: ...",
    "context7": "...optional extra context..."
  },
  ...
]

For each row we build an eval example where:
- input  = system prompt + user prompts (+ optional extra context)
- output = concatenated markdown file contents OR subagent_output

We then run an LLM judge to grade conciseness (0-1) with a short comment.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langsmith.evaluation import StringEvaluator, evaluate

logger = logging.getLogger(__name__)

_JUDGE: Optional[BaseChatModel] = None


def _get_judge() -> BaseChatModel:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = init_chat_model(model="gpt-5-mini", model_provider="openai", temperature=0)
    return _JUDGE


def _extract_text(resp: Any) -> str:
    """Best-effort stringify LLM responses."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c)
    return str(content)


def _call_judge(prompt: str, judge: Optional[BaseChatModel] = None) -> Dict[str, Any]:
    """Call the judge LLM and coerce a JSON-ish reply."""
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


def _conciseness_grader(inp: str, pred: str, _: str | None) -> Dict[str, Any]:
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


conciseness_eval = StringEvaluator(
    evaluation_name="conciseness",
    input_key="input",
    prediction_key="output",
    grading_function=_conciseness_grader,
)


def _coerce_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    return [str(val)]


def _load_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_examples(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
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

        examples.append(
            {
                "input": "\n\n".join(ctx_parts).strip(),
                "output": output,
                "subagent_name": row.get("subagent_name", "unknown"),
            }
        )
    return examples


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_langsmith_key() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def _normalize_comment(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _format_score(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    return f"{score:.2f}"


def _extract_eval_result(
    eval_results: Iterable[Any], key: str
) -> Tuple[Optional[float], Optional[str]]:
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
    rows: List[Dict[str, Any]] = []
    for row in results:
        example_inputs = getattr(row.get("example"), "inputs", {}) or {}
        subagent_name = example_inputs.get("subagent_name", "unknown")
        eval_results = (row.get("evaluation_results") or {}).get("results", [])
        score, comment = _extract_eval_result(eval_results, "conciseness")
        rows.append(
            {
                "subagent_name": subagent_name,
                "score": score,
                "comment": comment,
            }
        )
    return rows


def _run_local_eval(data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex in data:
        result = _conciseness_grader(ex["input"], ex["output"], None)
        rows.append(
            {
                "subagent_name": ex.get("subagent_name", "unknown"),
                "score": result.get("score"),
                "comment": result.get("comment"),
            }
        )
    return rows


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    summary = {
        "count": len(rows),
        "scored": len(scores),
        "missing": len(rows) - len(scores),
        "avg": None,
        "min": None,
        "max": None,
    }
    if scores:
        summary["avg"] = sum(scores) / len(scores)
        summary["min"] = min(scores)
        summary["max"] = max(scores)
    return summary


def _render_markdown_summary(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    dataset_path: str,
    experiment_name: Optional[str],
) -> str:
    lines = [
        "## Conciseness evals",
        f"Dataset: `{dataset_path}`",
        (
            "Examples: {count} (scored {scored}, missing {missing})".format(
                **summary
            )
        ),
    ]
    if experiment_name:
        lines.append(f"LangSmith experiment: `{experiment_name}`")
    if summary["avg"] is not None:
        lines.append(
            "Average score: {avg:.2f} (min {min:.2f}, max {max:.2f})".format(
                **summary
            )
        )
    lines.append("")
    lines.append("| Subagent | Score | Comment |")
    lines.append("| --- | --- | --- |")
    for row in rows:
        comment = _normalize_comment(row.get("comment"))
        comment = comment.replace("|", "\\|")
        lines.append(
            f"| {row.get('subagent_name', 'unknown')} | {_format_score(row.get('score'))} | {comment} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_github_summary(markdown: str) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(markdown)
        if not markdown.endswith("\n"):
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conciseness evals over subagent outputs.")
    parser.add_argument(
        "--history",
        required=True,
        help="Path to JSON history dataset.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="subagent-conciseness",
        help="Prefix for the LangSmith experiment name.",
    )
    parser.add_argument(
        "--langsmith",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whether to log results to LangSmith (auto=use if API key is set).",
    )
    args = parser.parse_args()

    if not _has_openai_key():
        raise SystemExit("OPENAI_API_KEY is required to run evals.")

    history = _load_history(args.history)
    if not history:
        raise SystemExit("No history entries found.")

    data = _build_examples(history)
    if not data:
        raise SystemExit("No eval examples built from history.")

    use_langsmith: bool
    if args.langsmith == "auto":
        use_langsmith = _has_langsmith_key()
    else:
        use_langsmith = args.langsmith == "on"
        if use_langsmith and not _has_langsmith_key():
            raise SystemExit("LANGCHAIN_API_KEY or LANGSMITH_API_KEY required for --langsmith on.")

    experiment_name: Optional[str] = None
    if use_langsmith:
        results = evaluate(
            lambda ex: {"output": ex["output"]},
            data=data,
            evaluators=[conciseness_eval],
            experiment_prefix=args.experiment_prefix,
            description="Conciseness of subagent outputs vs provided prompts.",
            metadata={"source": "local-subagent-history"},
            max_concurrency=0,
        )
        rows = _collect_langsmith_rows(results)
        experiment_name = getattr(results, "experiment_name", None)
    else:
        rows = _run_local_eval(data)

    summary = _summarize_rows(rows)

    print("Conciseness evals")
    print(f"Dataset: {args.history}")
    if experiment_name:
        print(f"LangSmith experiment: {experiment_name}")
    print(
        "Examples: {count} (scored {scored}, missing {missing})".format(**summary)
    )
    if summary["avg"] is not None:
        print(
            "Average score: {avg:.2f} (min {min:.2f}, max {max:.2f})".format(
                **summary
            )
        )
    for row in rows:
        comment = _normalize_comment(row.get("comment"))
        print(
            f"- {row.get('subagent_name', 'unknown')}: {_format_score(row.get('score'))} | {comment}"
        )

    markdown = _render_markdown_summary(summary, rows, args.history, experiment_name)
    _write_github_summary(markdown)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
