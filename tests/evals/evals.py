"""
Conciseness evaluation for subagent markdown outputs.

Expected input data (JSON file passed via --history):
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

For each row we build an eval example where:
- input  = system prompt + human prompts (the subagent's context)
- output = concatenated markdown file contents produced by that subagent

We then run a LangSmith StringEvaluator that grades conciseness (0–1) with a short comment.
"""

import argparse
import json
import logging
from typing import Any, Dict, List

from langchain.chat_models import init_chat_model
from langsmith import Client
from langsmith.evaluation import StringEvaluator, evaluate

logger = logging.getLogger(__name__)
client = Client()


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


def _call_judge(prompt: str) -> Dict[str, Any]:
    """Call the judge LLM and coerce a JSON-ish reply."""
    judge = init_chat_model(model="gpt-5-mini", model_provider="openai", temperature=0)
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
You score subagent markdown outputs for conciseness.
Context (system + human prompts):
{inp}

Subagent markdown output:
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
        user_messages = _coerce_list(row.get("user_messages"))
        md_files = row.get("md_files") or []
        md_chunks = []
        for f in md_files:
            name = f.get("name", "file")
            text = f.get("text", "")
            md_chunks.append(f"# {name}\n{text}".strip())

        ctx_parts = [
            f"System prompt:\n{row.get('system_prompt', '').strip()}",
            "User messages:\n" + "\n".join(user_messages),
        ]

        examples.append(
            {
                "input": "\n\n".join(ctx_parts).strip(),
                "output": "\n\n".join(md_chunks).strip(),
                "subagent_name": row.get("subagent_name", "unknown"),
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run conciseness evals over subagent markdown outputs.")
    parser.add_argument(
        "--history",
        required=True,
        help="Path to JSON history with subagent_name/system_prompt/user_messages/md_files fields.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="subagent-conciseness",
        help="Prefix for the LangSmith experiment name.",
    )
    args = parser.parse_args()

    history = _load_history(args.history)
    if not history:
        raise SystemExit("No history entries found.")

    data = _build_examples(history)

    results = evaluate(
        lambda ex: {"output": ex["output"]},
        data=data,
        evaluators=[conciseness_eval],
        experiment_prefix=args.experiment_prefix,
        description="Conciseness of subagent markdown outputs vs provided prompts.",
        metadata={"source": "local-subagent-history"},
    )
    print(f"View results in LangSmith: {results.experiment_url}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
