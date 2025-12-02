
from typing import Any


from langchain.agents import AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime
from src.schemas import ContextSchema


@dynamic_prompt
def context_middleware(request: ModelRequest) -> dict | None:  
    """Inject user context into the LLM input. (Access to user's resume etc)"""

    resume = f"""User's Resume:\n{request.runtime.context.model_dump() if request.runtime.context else None}"""

    return resume
