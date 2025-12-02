
from typing import Any


from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from UI.schemas import ContextSchema


@before_model
def context_middleware(state: AgentState, runtime: Runtime[ContextSchema]) -> dict | None:  
    """Inject user context into the LLM input. (Access to user's resume etc)"""

    return runtime.context.model_dump() if runtime.context else None
