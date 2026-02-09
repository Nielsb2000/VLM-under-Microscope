# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# (optional) non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# install uv
RUN python -m pip install --no-cache-dir uv

# copy dependency files first for cache
COPY pyproject.toml uv.lock* ./

# install deps (recommended: frozen if you have uv.lock)
RUN uv sync --frozen --no-dev

# copy app
COPY . .

EXPOSE 8000

USER appuser

# run via uv so it uses the env it created
CMD ["uv", "run", "gunicorn", "--bind=0.0.0.0:8000", "YOUR_MODULE:app"]
