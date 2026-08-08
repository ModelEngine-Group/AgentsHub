# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG INSTALL_NEURAL=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt requirements-docker-neural.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt \
    && if [ "${INSTALL_NEURAL}" = "true" ]; then \
         python -m pip install -r requirements-docker-neural.txt; \
       fi

COPY . .

RUN mkdir -p /app/data /app/outputs

LABEL org.opencontainers.image.title="MediGraph Agent" \
      org.opencontainers.image.description="MediGraph MCP and A2A runtime" \
      org.opencontainers.image.licenses="MIT"

CMD ["python", "-X", "utf8", "mcp_server/server.py"]
