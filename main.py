import os
from typing import Literal
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_tavily import TavilyExtract
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore  
from langgraph.graph.state import CompiledStateGraph
from langchain.tools import BaseTool
from typing import List, Optional, Any, Union, Dict
from uuid import uuid4
from langchain.chat_models.base import BaseChatModel
from prompts import deep_agent_prompt
import asyncio
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langsmith import uuid7
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper


load_dotenv()

class ContextSchema(BaseModel):
    role: str = Field(..., description="The job role the user is applying for")
    resume: str = Field(None, description="The user's resume or background information")
    experience_level: Literal["intern", "junior", "mid", "senior", "lead"] = Field(..., description="The user's experience level")
    years_of_experience: Optional[int] = Field(default=None, description="The number of years of experience the user has")

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

    

    async def _create_config(self) -> Dict[str, Any]:
        config = {
                "configurable": {
                    "thread_id": f"thread_{uuid7()}"
                    }
            }
        
        self.configs.append(config)

        return config

    async def _stream_response(
            self,
            user_input: str, 
            context: ContextSchema, 
            agent: CompiledStateGraph
            ) -> None:
        
        config = await self._create_config()
        
        async for message_chunk, metadata in agent.astream(
            {
                "messages": [
                    {"role": "user", "content": user_input + "Context\n" + context.model_dump_json()},
                ]
            },
            stream_mode="messages",
            config=config,
            ):
                if message_chunk.content:
                    print(message_chunk.content, end='|', flush=True)

    async def execute_agent(
            self,
            input: str,
            context: ContextSchema,
        ) -> None: 

        await self._stream_response(
            user_input=input,
            context=context,
            agent=self.agent,
            )

        

async def main():
    llm = init_chat_model(
        model="gpt-5",
        model_provider="openai",
        temperature=0,
    )

    search_tool = TavilySearch(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=5
    )

    search_tool_prompt = "Use `tavily_search` to find recent company/industry facts"
    
    
    extract_tool = TavilyExtract(
        api_key=os.getenv("TAVILY_API_KEY"),
        model="gpt-5-mini",
        temperature=0
    )

    extract_tool_prompt = "Use `tavily_extract` to pull specifics from result URLs. Use this tool after "

    reddit_tool = RedditSearchRun(
        api_wrapper=RedditSearchAPIWrapper(
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            reddit_user_agent="interview_helper",
        )
    )

    reddit_tool_prompt = """Use reddit_tool to search reddit. Think about every parameter before calling this tool."""

    tools = [search_tool, extract_tool, reddit_tool]

    tools_instructions = search_tool_prompt + "\n" + extract_tool_prompt + "\n" + reddit_tool_prompt

    user_input = "Help me to prepare for the interview https://careers.lg.com/apply/detail?id=1001099"

    context = ContextSchema(
        role="AI engineer",
        resume="""Ten Danil 
        LinkedIn| daniltenzxc123@gmail.com | GitHub 
        Languages: English, Korean, Russian 
        EDUCATION 
        Russian Technological University, MIREA 
        Bachelor of software engineering  
        Kyungnam University(Transfer) 
        Bachelor of computer engineering 
        PROJECTS 
        Moscow, Russia 
        2020 – 2022 
        Masan, Korea  
        2022 – 2025 
        Trip planner | Python, FastAPI, LangChain, LangGraph, LangSmith, git 
        Repository 
        09/2025 – Present 
        • Built a multi-agent itinerary planner with human-in-the-loop interrupts and short-term memory using LangGraph; 
        exposed via FastAPI with typed, structured outputs (Pydantic). Designed and built a multi-agent AI workflow 
        incorporating external tools, human-in-the-loop, interrupts, and short-term memory using LangGraph. 
        • Designed a RAG pipeline (chunking + embedding + re-rank) that reduced hallucinations and improved factual hit
        rate; added few-shot prompting for robustness.  
        • Parallelized agent graph execution to cut latency ~2× (180s → 80s) and stabilized outputs via JSON-schema 
        enforcement.  
        • Instrumented with LangSmith traces to debug tool calls and prompt drift. 
        Personal dictionary | Python, FastAPI, LangChain, LangGraph, LangSmith, Git, PostgreSQL  
        Repository 
        07/2025 – 09/2025 
        Shipped a dictionary app supporting translate/synonyms/example generation with Ollama-served Gemma; responses 
        schema-validated.  
        • Stored terms, embeddings, and usage examples in PostgreSQL + pgvector; synonym search powered by vector 
        similarity + lexical fallback. 
        • Tuned prompts and output structuring to lower failure modes (timeouts, invalid JSON) and improve perceived 
        quality. 
        Twitter disaster | Python, TensorFlow, Keras, scikit-learn, Pandas, matplotlib    
        Repository 
        06/2025 – 07/2025 
        • Fine-tuned BERT for disaster tweet classification on the standard ~10k-tweet dataset; built a reproducible 
        preprocessing pipeline (cleaning, tokenization, stratified split).  
        • Reached 84% accuracy. 
        Sneakers classification | Python, TensorFlow, Keras, scikit-learn, Pandas, matplotlib  
        Repository 
        05/2025 – 06/2025 
        • Fine-tuned ResNet50 with augmentation; 83% accuracy; analyzed error modes via confusion matrix.  
        • Reached 83% accuracy. 
        EXPERIENCE 
        CrewAI’s course alpha tester on Deeplearning.AI platform 
        Deeplearning.AI  
        08/2025 – Present 
        • As an alpha tester, I verified— from a real learner’s perspective—whether the curriculum and hands-on environment 
        worked properly, and proposed improvements. 
        • Ran the lecture content, labs, and quizzes end-to-end to check technical accuracy, difficulty, and clarity of the 
        guidance. 
        • Reported issues in a reproducible format—dead/inactive links, quiz typos and answer errors, explanations based on 
        an outdated CrewAI version, and unclear concept explanations. 
        TECHNICAL SKILLS 
        Languages: Python, SQL (Postgres) 
        Frameworks: Flask, FastAPI, TensorFlow, Keras 
        Developer Tools: Git, Docker, VS Code, Cursor, Codex, MCP 
        Libraries: LangChain, LangGraph, LangSmith, Pandas, Numpy, Matplotlib, scikit-learn""",
        experience_level="intern",
        years_of_experience=0
    )
    async with (
            AsyncPostgresStore.from_conn_string(os.getenv("DB_URL")) as store,
            AsyncPostgresSaver.from_conn_string(os.getenv("DB_URL")) as checkpointer,
        ):
            await store.setup()
            await checkpointer.setup()

            workflow = Workflow(
                llm=llm,
                tools=tools,
                tools_instructions=tools_instructions,
                store=store,
                checkpointer=checkpointer,
                subagents=[
                     {
                          "name": "research_agent",
                          "description": "Used to make researches",
                          "system_prompt": "You are a great researches. You should use tools to search information. Do not invent anything, check facts using tools.",
                          "tools": tools
                     }
                ]
            )
            await workflow.execute_agent(
                input=user_input,
                context=context
            )

if __name__ == "__main__": 
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)

    
