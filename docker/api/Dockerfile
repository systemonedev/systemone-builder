FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/systemone_builder /app/backend/systemone_builder
RUN pip install --no-cache-dir /app/backend

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["python", "-m", "systemone_builder.api.app"]
