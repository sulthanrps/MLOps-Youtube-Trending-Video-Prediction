# Layanan REST API inferensi youtube-trending-24h-classifier
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/models/production

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY models/production/model/requirements.txt /tmp/model-requirements.txt
COPY requirements-serving.txt /tmp/requirements-serving.txt
RUN grep -viE '^mlflow' /tmp/model-requirements.txt > /tmp/model-req.txt \
    && pip install -r /tmp/model-req.txt -r /tmp/requirements-serving.txt

COPY models/production/ ./models/production/
RUN test -f models/production/model/model.pkl \
    && test -f models/production/model_metadata.yaml \
    || (echo "ERROR: artefak model tidak ditemukan. Jalankan 'dvc pull models/production.dvc' sebelum build." && exit 1)

COPY app.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# host 0.0.0.0 agar layanan dapat diakses dari luar kontainer
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]