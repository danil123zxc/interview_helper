import json
import logging
import os
import time
from dotenv import load_dotenv
from langchain_tavily import TavilyExtract
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from src.schemas import ContextSchema
from src.workflow import Workflow
from src.logging_config import setup_logging, init_sentry
from src.db import workflow_ctx
load_dotenv()

setup_logging()
init_sentry()
logger = logging.getLogger(__name__)

def main():

    user_input = "Help me to prepare for the interview https://careers.kakao.com/jobs/P-14347?skillSet=&part=TECHNOLOGY&company=KAKAO&keyword=&employeeType=&page=1"

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
    logger.info("Starting workflow run with Postgres checkpointing")

    def _truncate_for_log(value, limit: int = 1200) -> str:
        if value is None:
            return ""
        try:
            text = json.dumps(value, default=str)
        except Exception:
            text = str(value)
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _log_final_state(wf: Workflow) -> None:
        try:
            state = wf.get_final_state()
        except Exception as exc:
            logger.warning("Failed to load final state: %s", exc, exc_info=True)
            return

        if isinstance(state, dict):
            state_obj = state.get("values") or state.get("state") or state
        else:
            state_obj = getattr(state, "values", None) or getattr(state, "state", None) or state

        state_keys = list(state_obj.keys()) if isinstance(state_obj, dict) else []
        md_files = []
        try:
            md_files = wf.list_md_files(wf.config)
        except Exception as exc:
            logger.warning("Failed to load markdown files from state: %s", exc, exc_info=True)

        file_names = [f.get("name") for f in md_files if isinstance(f, dict)]
        logger.info("Final state keys: %s", state_keys)
        logger.info("Final markdown files: %s", file_names)
        logger.debug("Final state summary: %s", _truncate_for_log(state_obj))

    try:
        with workflow_ctx() as workflow:
            start = time.monotonic()
            res = workflow.invoke(user_input, context=context, config=workflow.config)
            elapsed = time.monotonic() - start
            logger.info("Workflow run finished in %.2fs", elapsed)
            _log_final_state(workflow)
    except Exception as exc:
        logger.warning(
            "Postgres workflow failed, falling back to in-memory workflow: %s",
            exc,
            exc_info=True,
        )
        workflow = Workflow()
        start = time.monotonic()
        res = workflow.invoke(user_input, context=context, config=workflow.config)
        elapsed = time.monotonic() - start
        logger.info("Workflow run finished (in-memory) in %.2fs", elapsed)
        _log_final_state(workflow)

if __name__ == "__main__": 
    main()

    
    
