FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests
COPY scripts ./scripts

EXPOSE 8000

# The container is considered healthy only once the readiness probe reports
# that the adjudication endpoint has finished its startup self-check.
HEALTHCHECK --interval=3s --timeout=3s --start-period=2s --retries=20 \
    CMD ["python", "scripts/healthcheck.py"]

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
