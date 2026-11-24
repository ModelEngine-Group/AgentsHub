# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All rights reserved.
import base64
import os
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

MODEL = "BAAI/bge-small-zh-v1.5"
encoder = None

@asynccontextmanager
async def lifespan(app):
    global encoder
    encoder = TextEmbedding(model_name=MODEL, cache_dir="/embedding/models", threads=2)
    yield

app = FastAPI(lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"

@app.get("/health")
def health():
    return {"ready": encoder is not None, "model": MODEL}

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}

@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    if request.model != MODEL:
        raise HTTPException(404, "Unknown model")
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts or len(texts) > 32 or any(len(x) > 12000 for x in texts):
        raise HTTPException(400, "Use 1-32 texts, each at most 12000 characters")
    if request.encoding_format not in ("float", "base64"):
        raise HTTPException(400, "Unsupported encoding format")
    vectors = list(encoder.embed(texts, batch_size=8))
    data = []
    for i, vector in enumerate(vectors):
        value = base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode() if request.encoding_format == "base64" else vector.tolist()
        data.append({"object": "embedding", "index": i, "embedding": value})
    return {"object": "list", "model": MODEL, "data": data, "usage": {"prompt_tokens": 0, "total_tokens": 0}}
