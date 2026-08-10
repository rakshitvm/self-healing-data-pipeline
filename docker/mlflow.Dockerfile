# Minimal image for self-hosting the MLflow Tracking Server (Tier 1 local
# development only) against a PostgreSQL backend store.
#
# Versions pinned to match requirements.txt exactly. psycopg2-binary is
# required so SQLAlchemy (used internally by MLflow's backend store) can
# speak to the postgresql:// backend-store-uri configured in
# docker-compose.yml.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mlflow==3.15.1 \
    psycopg2-binary==2.9.12

WORKDIR /mlflow
