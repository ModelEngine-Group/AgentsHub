FROM python:3.12-slim

WORKDIR /app
COPY travel_mcp ./travel_mcp

ENV PYTHONUNBUFFERED=1
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).status==200 else 1)"

CMD ["python", "-m", "travel_mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
