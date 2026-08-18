FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
EXPOSE 18088
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python -c "import urllib.request; raise SystemExit(0 if urllib.request.urlopen('http://127.0.0.1:18088/health', timeout=3).status < 400 else 1)"
CMD ["uvicorn", "tool_server.app:app", "--host", "0.0.0.0", "--port", "18088"]
