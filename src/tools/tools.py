from langchain_tavily import TavilySearch, TavilyExtract
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
import os
from dotenv import load_dotenv

from src.tools.pdf_tool import load_pdf_tool


load_dotenv()


def build_tools():
    tavily_search = TavilySearch(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=3,
    )
    tavily_extract = TavilyExtract(
        api_key=os.getenv("TAVILY_API_KEY"),
        model="gpt-5-mini",
        temperature=0,
    )
    tools = {
            'tavily_search':tavily_search,
            'tavily_extract': tavily_extract,
            'load_pdf_tool': load_pdf_tool
             }

    instructions = [
        "Use `tavily_search` to find recent company/industry facts",
        "Use `tavily_extract` to pull specifics from result URLs. Use this tool after search results.",
    ]

    reddit_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if reddit_id and reddit_secret:
        try:
            reddit_search = RedditSearchRun(
                api_wrapper=RedditSearchAPIWrapper(
                    reddit_client_id=reddit_id,
                    reddit_client_secret=reddit_secret,
                    reddit_user_agent="interview_helper",
                )
            )
            tools['reddit_search'] = reddit_search
            instructions.append("Use `reddit_tool` to search Reddit. Think about every parameter before calling this tool.")
        except Exception:
            pass

    tools_instructions = "\n".join(instructions)

    return tools, tools_instructions
