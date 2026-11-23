FROM python:3.12-slim
WORKDIR /app
COPY mcps/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY mcps/server.py mcps/retail_embeddings.py ./mcps/
COPY data/orders.csv ./data/orders.csv
ENV MCP_HOST=0.0.0.0 PYTHONUNBUFFERED=1
USER 65534:65534
EXPOSE 8765
CMD ["python", "mcps/server.py"]
