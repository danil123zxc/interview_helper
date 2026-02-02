"""Shared constants for evals."""

from typing import Dict, List

from src.prompts import (
    analyze_agent_prompt,
    job_posting_ingestor_prompt,
    planner_agent_prompt,
    question_writer_prompt,
    research_agent_prompt,
    synthesis_agent_prompt,
)

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
