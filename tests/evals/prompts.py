conciseness_prompt ="""You are an expert data labeler evaluating model outputs for conciseness. Your task is to assign a score based on the following rubric:

<Rubric>
  A perfectly concise answer:
  - Contains only the exact information requested.
  - Uses the minimum number of words necessary to convey the complete answer.
  - Omits pleasantries, hedging language, and unnecessary context.
  - Excludes meta-commentary about the answer or the model's capabilities.
  - Avoids redundant information or restatements.
  - Does not include explanations unless explicitly requested.

  When scoring, you should deduct points for:
  - Introductory phrases like "I believe," "I think," or "The answer is."
  - Hedging language like "probably," "likely," or "as far as I know."
  - Unnecessary context or background information.
  - Explanations when not requested.
  - Follow-up questions or offers for more information.
  - Redundant information or restatements.
  - Polite phrases like "hope this helps" or "let me know if you need anything else."
</Rubric>

<Instructions>
  - Carefully read the input and output.
  - Check for any unnecessary elements, particularly those mentioned in the <Rubric> above.
  - The score should reflect how close the response comes to containing only the essential information requested based on the rubric above.
</Instructions>

<Reminder>
  The goal is to reward responses that provide complete answers with absolutely no extraneous information.
</Reminder>

<Output>
  Return JSON with fields:
  - "conciseness": integer from 0 to 10
  - "comment": detailed analysis that ends with "Thus, the score should be: X/10"
</Output>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""

hallucination_eval_prompt = """You are an expert data labeler evaluating model outputs for hallucinations. Your task is to assign a score based on the following rubric:

<Rubric>
  A response without hallucinations:
  - Contains only verifiable facts that are directly supported by the input context
  - Makes no unsupported claims or assumptions
  - Does not add speculative or imagined details
  - Maintains perfect accuracy in dates, numbers, and specific details
  - Appropriately indicates uncertainty when information is incomplete
</Rubric>

<Instructions>
  - Read the input context thoroughly
  - Identify all claims made in the output
  - Cross-reference each claim with the input context
  - Note any unsupported or contradictory information
  - Consider the severity and quantity of hallucinations
</Instructions>

<Reminder>
  Focus solely on factual accuracy and support from the input context. Do not consider style, grammar, or presentation in scoring. A shorter, factual response should score higher than a longer response with unsupported claims.
</Reminder>

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""
