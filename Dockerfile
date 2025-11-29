# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app

# Install dependencies (no project files yet to maximize cache hits)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Install project into the virtualenv
COPY . .
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY --from=builder /app/.venv /app/.venv
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "UI/streamlit_app.py"]
