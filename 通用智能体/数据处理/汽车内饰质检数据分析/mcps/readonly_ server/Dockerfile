FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Demo data is baked in; mount /app/data to use your own detection.db.
ENV PYTHONUNBUFFERED=1
EXPOSE 5071
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "5071", "--db-path", "/app/data/demo.db", "--knowledge-dir", "/app/knowledge"]
