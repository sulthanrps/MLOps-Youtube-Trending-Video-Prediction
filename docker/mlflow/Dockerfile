FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install \
    mlflow==2.22.5 \
    "sqlalchemy>=2.0,<2.1" \
    psycopg2-binary==2.9.10 \
    PyYAML==6.0.2

WORKDIR /app
EXPOSE 5000