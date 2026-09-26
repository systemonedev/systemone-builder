FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY README.md /app/README.md
COPY backend/systemone /app/backend/systemone
RUN pip install --no-cache-dir /app/backend

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["python", "-m", "systemone.api.app"]
