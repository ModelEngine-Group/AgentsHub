FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python -c "import urllib.request; raise SystemExit(0 if urllib.request.urlopen('http://127.0.0.1:8501', timeout=3).status < 400 else 1)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
