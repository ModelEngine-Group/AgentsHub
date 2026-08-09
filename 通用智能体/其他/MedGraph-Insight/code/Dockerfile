FROM python:3.11-slim

# 中文字体（matplotlib 可视化需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e .

# 默认运行端到端全链路 Demo
CMD ["python", "-m", "mgi.cli", "pipeline"]
