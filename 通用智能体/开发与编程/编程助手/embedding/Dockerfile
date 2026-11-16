FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
ENV OMP_NUM_THREADS=2
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8766"]
