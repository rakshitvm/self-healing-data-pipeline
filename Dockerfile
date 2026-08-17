# Self-Healing Data Pipeline — application image.
#
# Packages the real Tier 1 / Tier 2 CLI application under src/, exactly
# as it already runs on the host: this project's pyproject.toml has no
# [project]/[build-system] table (see pyproject.toml — it's intentionally
# empty), so there is no `pip install .` packaging step anywhere in this
# project's real workflow. The application has always been run as
# `PYTHONPATH=src python -m self_healing_pipeline.interfaces.cli.main
# ...` (see README.md, scripts/); this image reproduces that exact
# invocation via PYTHONPATH, unmodified.
#
# PostgreSQL, MLflow, Elasticsearch, Kibana, and Filebeat are external
# services provisioned by docker-compose.yml — this image contains only
# the application itself and reaches them over the network via the same
# environment variables the host CLI already uses (see .env.example).
# No secrets are declared, baked in, or defaulted here; all
# configuration comes from the container's runtime environment.

FROM python:3.12-slim

WORKDIR /app

# --- Dependencies first (layer-cache friendly): only requirements.txt
# (runtime deps) is ever installed here. requirements-dev.txt (pytest,
# ruff, mypy, black, pre-commit, ...) is intentionally never installed
# into this image. ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# --- Non-root runtime user ---
RUN useradd --create-home --uid 1000 --user-group appuser \
    && chown appuser:appuser /app

# --- Application source, copied last so dependency layers above stay
# cached across ordinary source-code changes. ---
COPY --chown=appuser:appuser src/ /app/src/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# No HEALTHCHECK: `repair`/`schema-repair` are one-shot CLI invocations
# (the container runs one command against one file/table and exits),
# not a long-running daemon or HTTP service. There is no process to
# periodically probe, so a HEALTHCHECK here would either sit unused
# (the container has usually already exited before the first interval
# fires) or require inventing an HTTP endpoint this application does
# not have — explicitly out of scope.

ENTRYPOINT ["python", "-m", "self_healing_pipeline.interfaces.cli.main"]
CMD ["--help"]
