from typing import Annotated

from pydantic import BaseModel, Field


class ConcisenessEval(BaseModel):
    """Structured output schema for conciseness evaluation.

    Args:
        comment: Detailed analysis ending with "Thus, the score should be: X/10".
        conciseness: Integer score from 0 to 10.

    Returns:
        ConcisenessEval instance with validated fields.

    Example:
        ```python
        eval_obj = ConcisenessEval(
            comment="Concise overall. Thus, the score should be: 8/10",
            conciseness=8,
        )
        eval_obj.model_dump()
        # {"comment": "...", "conciseness": 8}
        ```
    """

    comment: str = Field(
        ...,
        description=(
            "A detailed analysis that: 1) Identifies any unnecessary elements "
            "(hedging, pleasantries, meta-commentary, etc.), 2) Notes any redundant "
            "or extraneous information, 3) Evaluates if explanations were explicitly "
            "requested, 4) Analyzes word efficiency, and 5) Ends with "
            "\"Thus, the score should be: X/10\" where X reflects how close the "
            "response comes to perfect conciseness."
        ),
    )
    conciseness: Annotated[
        int,
        Field(
            ge=0,
            le=10,
            description=(
                "Evaluates how efficiently the output conveys the required information without "
                "any unnecessary elements. Min (0): Extremely verbose with many unnecessary "
                "elements. Max (10): Contains only essential requested information with no "
                "extra words."
            ),
        ),
    ]


class HallucinationEval(BaseModel):
    """Structured output schema for hallucination evaluation.

    Args:
        comment: Analysis ending with "Thus, the score should be: TRUE/FALSE".
        hallucination: True if hallucinations are present, otherwise False.

    Returns:
        HallucinationEval instance with validated fields.

    Example:
        ```python
        eval_obj = HallucinationEval(
            comment="All claims are supported. Thus, the score should be: FALSE",
            hallucination=False,
        )
        eval_obj.model_dump()
        # {"comment": "...", "hallucination": False}
        ```
    """

    comment: str = Field(
        ...,
        description=(
            "A detailed analysis that: 1) Lists any claims made in the output, 2) "
            "Identifies which claims are supported/unsupported by the input context, "
            "3) Notes any contradictions or speculative additions, and 4) Ends with "
            "\"Thus, the score should be: TRUE/FALSE\" based on whether any hallucinations "
            "were found."
        ),
    )
    hallucination: bool = Field(
        ...,
        description=(
            "TRUE if the output contains any hallucinations (unsupported claims, contradictions, "
            "speculative details, or inaccurate facts). FALSE if all claims are directly "
            "verifiable from the input context."
        ),
    )
