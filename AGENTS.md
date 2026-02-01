# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core Python modules (`workflow.py`, `main.py`, `prompts.py`, `schemas.py`, `tools/`).
- `UI/`: Streamlit frontend (`streamlit_app.py` and `pages/`).
- `tests/`: pytest suite; `tests/evals/` holds evaluation helpers.
- `data/`: local artifacts or saved outputs.
- `docker-compose.yml`, `Dockerfile`: containerized app + Postgres.
- `.github/workflows/`: CI/CD pipelines.

## Build, Test, and Development Commands
- `uv sync`: install dependencies from `pyproject.toml`/`uv.lock`.
- `uv run pytest -q`: run the full test suite (matches CI).
- `python src/main.py`: run the agent workflow (streams output to stdout).
- `streamlit run UI/streamlit_app.py`: launch the web UI locally.
- `docker compose up --build`: start app + Postgres with Docker.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, and PEP 8 style.
- Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Keep line length reasonable (~100) and follow patterns in `src/`.
- No formatter/linter is enforced; keep imports tidy and add type hints where practical.

## Testing Guidelines
- Framework: pytest (`tests/`, with mocks/stubs for external tools).
- Name tests `test_*.py` and test functions `test_*`.
- Prefer deterministic tests (see `tests/test_workflow.py` for stubbing patterns).
- Run a single test: `uv run pytest tests/test_workflow.py -k <name>`.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, sentence-case (e.g., “Add …”, “Refactor …”).
- Keep subjects concise and avoid noisy prefixes unless required by a tool.
- PRs should include: purpose, key changes, test notes, and screenshots/GIFs for Streamlit UI changes. Link related issues if applicable.

## Configuration & Secrets
- Local configuration lives in `.env`; never commit real API keys.
- Required env vars: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `REDDIT_*`, `DB_URL` (see `README.md`).


## MCP Usage
- Always use Context7 if you need any library docs(ex. langchain, langgraph, etc) 