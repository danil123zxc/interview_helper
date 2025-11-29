# Interview Helper

LangGraph + DeepAgents workflow that prepares you for a specific job interview by researching the company/role and generating tailored practice material. It streams responses from an OpenAI model while persisting state in Postgres.

## What it does
- Builds a deep agent graph with a research subagent for web and Reddit lookups.
- Uses Tavily Search/Extract to gather fresh company info and RedditSearch to collect peer tips.
- Streams structured prep output (plan, practice Q&A, gap analysis, checklist) based on `prompts.py`.
- Stores graph state/checkpoints in Postgres so runs can be resumed/inspected.

## Requirements
- Python 3.12+
- Access to: OpenAI API, Tavily API, Reddit API, Postgres (reachable via `DB_URL`)
- Optional: LangSmith for tracing (`LANGCHAIN_TRACING_V2=true`).

## Setup
1) Create a `.env` (replace placeholders below; rotate any committed secrets):
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...        # only if using LangSmith
LANGCHAIN_PROJECT=interview-helper
DB_URL=postgresql://user:pass@host:5432/interview_helper?sslmode=disable
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```
2) Install deps (choose one):
```
uv sync          # preferred, uses pyproject/uv.lock
# or
pip install -e .
```
3) Ensure Postgres database in `DB_URL` exists and is reachable.

## Run the workflow
```
python main.py
```
`main.py` currently seeds a sample `user_input` and `ContextSchema` (role, resume, experience). Edit those values to match your interview. Output streams to stdout, chunk-delimited by `|`.

## How it works
- `Workflow` (main.py) builds the agent with tools and optional subagents; state/checkpoints are backed by `PostgresStore/PostgresSaver`.
- Prompting lives in `prompts.py` (`deep_agent_prompt`).
- Tools wired in `main.py`: `TavilySearch`, `TavilyExtract`, and `RedditSearchRun`.
- Temporary helpers: `tmp_parse.py` (HTML parsing), `tmp_pginspect.py` (inspect LangGraph Postgres classes).

## Troubleshooting
- Permission issues pushing to GitHub: switch remote to HTTPS or add an SSH key.
- Postgres connection errors: verify `DB_URL`, DB exists, and sslmode matches your server.
- Tavily/Reddit calls failing: double-check API keys and network access.

## Next steps
- Add CLI/HTTP interface instead of hardcoded context.
- Add `.env.example` and rotate any sensitive keys already committed.
- Write tests around the LangGraph workflow and tool wiring.
