"""
LangSmith evaluation for subagent outputs.

Scores each subagent output on:
- conciseness: brevity and on-topic quality
- hallucination: unsupported or contradictory statements vs provided context

Inputs are expected as a JSON list where each item has:
- subagent_name: name of the subagent (for metadata; not required by evaluators)
- system_prompt: the subagent's system prompt
- user_prompt: the human/user prompt passed to the subagent
- subagent_output: the subagent's reply to score
- context7: optional snippet from LangSmith docs or other context to ground hallucination checks

Example usage:
    python -m src.evals.subagent_eval --data data/subagent_outputs.sample.json --experiment-prefix subagent-evals
Requires LANGCHAIN/OPENAI credentials for the judge model.
"""

import argparse
import json
import logging
from typing import Any, Dict, List

from langchain.chat_models import init_chat_model
from langsmith.evaluation import StringEvaluator, evaluate

logger = logging.getLogger(__name__)


def _extract_text(resp: Any) -> str:
    """Best-effort stringify LLM responses."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        # LangChain messages sometimes return list segments
        return " ".join(str(c) for c in content if c)
    return str(content)


def _call_judge(prompt: str) -> Dict[str, Any]:
    """Call the judge LLM and coerce a JSON-ish reply."""
    # Keep deterministic for evals
    judge = init_chat_model(model="gpt-5-mini", model_provider="openai", temperature=0)
    raw = judge.invoke(prompt)
    text = _extract_text(raw).strip()

    try:
        parsed = json.loads(text)
    except Exception:
        # Fallback: try to pull score/comment heuristically
        parsed = {}
        lower = text.lower()
        if "score" in lower:
            # naive extraction of number between 0 and 1
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
    return {
        "score": score,
        "comment": parsed.get("comment", text),
    }


def _conciseness_grader(inp: str, pred: str, _: str | None) -> Dict[str, Any]:
    prompt = f"""
You score subagent answers for conciseness.
Grounding context (system prompt + user prompt + optional docs):
{inp}

Subagent output:
{pred}

Return JSON: {{"score": number between 0 and 1, "comment": "brief reason + suggested trim"}}.
Score 1 = fully concise, on-topic. Score 0 = verbose, repetitive, or off-task.
"""
    result = _call_judge(prompt)
    result["key"] = "conciseness"
    return result


def _hallucination_grader(inp: str, pred: str, _: str | None) -> Dict[str, Any]:
    prompt = f"""
You score hallucination risk using ONLY the provided context (system prompt, user prompt, and optional docs).
Grounding context:
{inp}

Subagent output:
{pred}

Flag statements not supported by the context or that contradict it.
Return JSON: {{"score": number between 0 and 1, "comment": "reason; list unsupported claims"}}.
Score 1 = no unsupported claims; 0 = largely hallucinated.
"""
    result = _call_judge(prompt)
    result["key"] = "hallucination"
    return result


conciseness_eval = StringEvaluator(
    evaluation_name="conciseness",
    input_key="input",
    prediction_key="subagent_output",
    grading_function=_conciseness_grader,
)

hallucination_eval = StringEvaluator(
    evaluation_name="hallucination",
    input_key="input",
    prediction_key="subagent_output",
    grading_function=_hallucination_grader,
)


def _load_examples(path: str) -> List[Dict[str, str]]:
    """Load JSON list of subagent records into eval-ready dicts."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples: List[Dict[str, str]] = []
    for row in raw:
        ctx_parts = [
            f"System prompt:\n{row.get('system_prompt', '').strip()}",
            f"User prompt:\n{row.get('user_prompt', '').strip()}",
        ]
        if row.get("context7"):
            ctx_parts.append(f"Docs context:\n{row['context7']}")
        examples.append(
            {
                "input": "\n\n".join(ctx_parts),
                "subagent_output": row.get("subagent_output", "").strip(),
                "subagent_name": row.get("subagent_name", "unknown"),
            }
        )
    return examples


def _identity_model(example: Dict[str, str]) -> Dict[str, str]:
    """Eval target: no new generation, just echo the provided subagent output."""
    return {"subagent_output": example["subagent_output"]}


def main():
    parser = argparse.ArgumentParser(description="Run LangSmith evals for subagents.")
    parser.add_argument("--data", required=True, help="Path to JSON list of subagent logs.")
    parser.add_argument(
        "--experiment-prefix",
        default="subagent-evals",
        help="Prefix for the LangSmith experiment name.",
    )
    args = parser.parse_args()

    data = _load_examples(args.data)
    if not data:
        raise SystemExit("No examples loaded from data file.")

    results = evaluate(
        _identity_model,
        data=data,
        evaluators=[conciseness_eval, hallucination_eval],
        experiment_prefix=args.experiment_prefix,
        description="Conciseness + hallucination evals for subagents.",
        metadata={"source": "local-subagent-outputs"},
    )
    print(f"View results in LangSmith: {results.experiment_url}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
