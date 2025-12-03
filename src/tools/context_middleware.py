
from typing import Any


from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from src.schemas import ContextSchema
from langchain_core.messages import HumanMessage

@before_model
def context_middleware(state: AgentState, runtime: Runtime[ContextSchema]) -> dict | None:  
    """Inject user context into the LLM input. (Access to user's resume etc)"""

    resume = f"""User's Resume:\n{runtime.context.model_dump() if runtime.context else None}"""

    return {"messages": [HumanMessage(content=resume)]}
