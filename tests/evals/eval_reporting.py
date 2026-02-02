"""Summary rendering for evals."""

import os
from typing import Any, Dict, List, Optional

from tests.evals.eval_constants import METRICS


def summarize_by_subagent(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute per-subagent summary stats for each metric.

    Args:
        rows: List of row dicts with subagent_name and metric score fields.

    Returns:
        Nested mapping of subagent -> metric key -> stats dict (count/avg/min/max).

    Example:
        ```python
        rows = [
            {"subagent_name": "analyze_agent", "conciseness_score": 8, "hallucination_score": 0},
            {"subagent_name": "analyze_agent", "conciseness_score": 6, "hallucination_score": 1},
        ]
        summary = summarize_by_subagent(rows)
        # summary["analyze_agent"]["conciseness"]["avg"] == 7.0
        # summary["analyze_agent"]["hallucination_rate"]["count"] == 2
        ```
    """
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


def normalize_comment(text: Optional[str], limit: int = 160) -> str:
    """Normalize whitespace and truncate comments for tables.

    Args:
        text: Comment text to normalize.
        limit: Maximum length of the returned string.

    Returns:
        Cleaned and optionally truncated string (empty if text is None/empty).

    Example:
        ```python
        normalize_comment("Hello   world", limit=20)
        # "Hello world"
        ```
    """
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def format_score(score: Optional[float]) -> str:
    """Format a numeric score for display.

    Args:
        score: Optional numeric score.

    Returns:
        "N/A" when score is None, otherwise a 2-decimal string.

    Example:
        ```python
        format_score(1.234)
        # "1.23"
        ```
    """
    if score is None:
        return "N/A"
    return f"{score:.2f}"


def render_markdown_summary(
    summary: Dict[str, Dict[str, Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    dataset_label: str,
) -> str:
    """Render a GitHub Actions summary for eval results.

    Args:
        summary: Aggregated stats from summarize_by_subagent.
        rows: Raw row results to populate the example table.
        dataset_label: Label shown in the summary header.

    Returns:
        Markdown string suitable for GitHub Actions step summary.

    Example:
        ```python
        summary = {
            "analyze_agent": {
                "conciseness": {"avg": 7.0},
                "hallucination_rate": {"avg": 0.0},
            }
        }
        rows = [
            {
                "example_id": "0",
                "subagent_name": "analyze_agent",
                "conciseness_score": 7,
                "hallucination_score": 0,
                "conciseness_comment": "ok",
                "hallucination_comment": "none",
            }
        ]
        md = render_markdown_summary(summary, rows, "dataset/dataset.py")
        # md.splitlines()[0] == "## Subagent evals"
        ```
    """
    lines = ["## Subagent evals", f"Dataset: `{dataset_label}`", ""]
    for subagent, metrics in summary.items():
        lines.append(f"### {subagent}")
        for metric in METRICS:
            stats = metrics.get(metric["key"], {})
            if metric["direction"] == "higher_better":
                direction = "higher is better"
            else:
                direction = "lower is better"
            lines.append(f"- {metric['label']} ({direction}): avg={format_score(stats.get('avg'))}")
        lines.append("")

    lines.append(
        "| Example | Subagent | Conciseness | Hallucination rate | "
        "Conciseness comment | Hallucination comment |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        conc_comment = normalize_comment(row.get("conciseness_comment"))
        hall_comment = normalize_comment(row.get("hallucination_comment"))
        conc_comment = conc_comment.replace("|", "\\|")
        hall_comment = hall_comment.replace("|", "\\|")
        lines.append(
            "| {example_id} | {subagent} | {conc} | {hall} | {conc_comment} | {hall_comment} |".format(
                example_id=row.get("example_id", "-"),
                subagent=row.get("subagent_name", "unknown"),
                conc=format_score(row.get("conciseness_score")),
                hall=format_score(row.get("hallucination_score")),
                conc_comment=conc_comment,
                hall_comment=hall_comment,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_github_summary(markdown: str) -> None:
    """Append markdown to the GitHub Actions step summary if available.

    Args:
        markdown: Markdown string to append.

    Returns:
        None. If GITHUB_STEP_SUMMARY is not set, this is a no-op.

    Example:
        ```python
        import os
        os.environ["GITHUB_STEP_SUMMARY"] = "summary.md"
        write_github_summary("## Results")
        # summary.md now contains "## Results"
        ```
    """
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(markdown)
        if not markdown.endswith("\n"):
            f.write("\n")
