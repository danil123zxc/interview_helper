
import logging

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph
from langchain.tools import BaseTool
from typing import List, Optional, Any, Union, Dict
from langchain.chat_models.base import BaseChatModel
from prompts import deep_agent_prompt

from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langsmith import uuid7
from schemas import ContextSchema

logger = logging.getLogger(__name__)

class Workflow:
    def __init__(self, 
                 llm: Optional[BaseChatModel]=None, 
                 system_prompt: Optional[str]=None, 
                 tools: Optional[Union[List[BaseTool], BaseTool]]=None,
                 tools_instructions: Optional[str]="",
                 store: Optional[BaseStore] = None, 
                 checkpointer: Optional[BaseCheckpointSaver] = None,
                 subagents: Optional[Dict[str, Any]] = None
                 ):
      
        self.llm = llm if llm else init_chat_model(
            model="gpt-5-mini",
            model_provider="openai",
            temperature=0,
        )

        self.system_prompt = system_prompt if system_prompt else deep_agent_prompt.format(
            tools_instructions=tools_instructions
        )
        self.tools = tools
        self.configs: List[Dict[str, Any]] = []
        self.store = store
        self.checkpointer = checkpointer

        self.agent = create_deep_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            store=self.store,
            subagents=subagents
        )
        logger.info("Deep agent created with %d tool(s); subagents=%s", len(self.tools or []), bool(subagents))

    async def _create_config(self) -> Dict[str, Any]:
        config = {
                "configurable": {
                    "thread_id": f"thread_{uuid7()}"
                    }
            }
        
        self.configs.append(config)

        return config
    

    async def stream_all(
            self,
            user_input: str,
            context: ContextSchema
        ):
        """Yield message chunks as the agent streams them."""
        config = await self._create_config()
        logger.debug("Starting stream with thread_id=%s", config["configurable"]["thread_id"])

        async for message_chunk, metadata in self.agent.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input + "Context\n" + context.model_dump_json(),
                    },
                ]
            },
            stream_mode="messages",
            config=config,
        ):
            role = getattr(message_chunk, "type", getattr(message_chunk, "role", "unknown"))
            content = message_chunk.content
            if isinstance(content, list):
                content = " ".join(str(c) for c in content if c)
            snippet = (content or "")[:500].replace("\n", " ")
            logger.debug("Chunk role:\n%s\n Content:\n%s", role, snippet)
            yield message_chunk

    async def stream_content(
                self,
                user_input: str,
                context: ContextSchema
            ):
        """Yield content (any role) as strings."""
        async for chunk in self.stream_all(
            user_input=user_input,
            context=context,
        ):
            yield chunk.content


    async def stream_ai_response(
                self,
                user_input: str,
                context: ContextSchema
            ):
        """Yield only the final combined AI message (not intermediate chunks)."""
        ai_chunks: List[str] = []
        async for chunk in self.stream_all(
            user_input=user_input,
            context=context,
        ):
            role = getattr(chunk, "type", getattr(chunk, "role", "unknown")).lower()
            if chunk.content and role == "ai":
                content = chunk.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content if c)
                ai_chunks.append(str(content))
        if ai_chunks:
            yield "".join(ai_chunks)

    async def _stream_response(
            self,
            user_input: str, 
            context: ContextSchema, 
            agent: CompiledStateGraph
            ) -> None:
        async for chunk in self.stream_ai_response(
            user_input=user_input,
            context=context,
            agent=agent,
        ):
            print(chunk, end='|', flush=True)


