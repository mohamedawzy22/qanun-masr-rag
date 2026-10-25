# syntax=docker/dockerfile:1.7

# ----------------------------------------------------------
# Builder: install locked production dependencies.
# Dependency installation is cached separately from source.
# ----------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,id=qanun-uv,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-install-project --no-dev


# ----------------------------------------------------------
# Runtime: only Python, installed dependencies, and app code.
# ----------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app \
        --home-dir /home/app --create-home app \
    && mkdir -p /app/src \
    && chown -R 10001:10001 /app /home/app

COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --chown=10001:10001 src/ /app/src/

USER 10001:10001

EXPOSE 8001

CMD ["uvicorn", "main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8001"]
