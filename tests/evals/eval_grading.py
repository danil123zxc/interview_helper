"""LLM-based graders for evals."""

import json
import logging
from typing import Any, Dict, Optional, Type

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel

from tests.evals.models import ConcisenessEval, HallucinationEval
from tests.evals.prompts import conciseness_prompt, hallucination_eval_prompt

logger = logging.getLogger(__name__)


def get_judge() -> BaseChatModel:
    """Create and return the judge chat model.

    Args:
        None.

    Returns:
        A BaseChatModel instance configured for evaluation.

    Example:
        ```python
        judge = get_judge()
        # isinstance(judge, BaseChatModel) is True
        ```
    """
    _JUDGE = init_chat_model(model="gpt-5-mini", model_provider="openai", temperature=0.7)
    return _JUDGE


def extract_text(resp: Any) -> str:
    """Convert an LLM response object into a plain string.

    Args:
        resp: LLM response object, string, or object with a "content" attribute.

    Returns:
        String representation of the response content.

    Example:
        ```python
        class Resp:
            content = ["Hello", "world"]

        extract_text(Resp())
        # "Hello world"
        ```
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c)
    return str(content)


async def call_judge_async(
    prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    judge: Optional[BaseChatModel] = None,
) -> Dict[str, Any]:
    """Call the judge LLM and parse structured or JSON-ish output.

    Args:
        prompt: Prompt string sent to the judge model.
        response_model: Optional Pydantic model for structured output parsing.
        judge: Optional pre-initialized chat model; defaults to get_judge().

    Returns:
        Parsed response dict. Includes at least a "comment" field and, when
        possible, a score field (e.g., "conciseness" or "score").

    Example:
        ```python
        result = await call_judge_async(
            "Score this output",
            response_model=ConcisenessEval,
        )
        # result["comment"] contains a textual rationale
        # result may include "conciseness" or "score"
        ```
    """
    judge = judge or get_judge()
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
    text = extract_text(raw).strip()

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


async def conciseness_grader_async(inp: str, pred: str) -> Dict[str, Any]:
    """Prompt the judge to score conciseness for a given input/output pair.

    Args:
        inp: Input text shown to the subagent (history or prompt).
        pred: Output text produced by the subagent (markdown files).

    Returns:
        Dict with keys: key ("conciseness"), score (float or None), comment (str).

    Example:
        ```python
        result = await conciseness_grader_async("Input", "Output")
        # result == {"key": "conciseness", "score": 8.0, "comment": "..."}
        ```
    """
    prompt = conciseness_prompt.format(inputs=inp, outputs=pred)
    result = await call_judge_async(prompt, response_model=ConcisenessEval)
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


async def hallucination_grader_async(inp: str, pred: str) -> Dict[str, Any]:
    """Prompt the judge to score hallucination rate for a given input/output pair.

    Args:
        inp: Input text shown to the subagent (history or prompt).
        pred: Output text produced by the subagent (markdown files).

    Returns:
        Dict with keys: key ("hallucination_rate"), score (float or None), comment (str).

    Example:
        ```python
        result = await hallucination_grader_async("Input", "Output")
        # result == {"key": "hallucination_rate", "score": 0.0, "comment": "..."}
        ```
    """
    prompt = hallucination_eval_prompt.format(inputs=inp, outputs=pred)
    result = await call_judge_async(prompt, response_model=HallucinationEval)
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
