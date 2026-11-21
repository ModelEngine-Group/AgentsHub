"""Local embedding and read-only math tools for the Nexent course assistant."""

import math
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

MODEL = "bge-small-zh-v1.5"
mcp = FastMCP(
    "course_math",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["nexent-math-course:8765", "127.0.0.1:8765"],
        allowed_origins=[],
    ),
)


@mcp.tool()
def quadratic_check(a: int, b: int, c: int, x: int | None = None) -> dict:
    """核对整数系数二次式 ax²+bx+c。给出判别式、顶点、实根及可选整数点的函数值。"""
    if a == 0 or any(abs(value) > 100000 for value in (a, b, c, x or 0)):
        return {"error": "a 必须非零，所有输入的绝对值须不超过 100000"}
    delta = b * b - 4 * a * c
    result = {
        "discriminant": delta,
        "vertex_x": str(Fraction(-b, 2 * a)),
        "vertex_y": str(Fraction(-delta, 4 * a)),
        "opens": "up" if a > 0 else "down",
        "roots": [],
    }
    if delta >= 0:
        square = math.isqrt(delta)
        if square * square == delta:
            result["roots"] = [
                str(root) for root in sorted({Fraction(-b - square, 2 * a), Fraction(-b + square, 2 * a)})
            ]
        else:
            result["roots"] = [f"({-b} ± √{delta}) / ({2 * a})"]
    if x is not None:
        result["value_at_x"] = a * x * x + b * x + c
    return result


@mcp.tool()
def fraction_check(
    numerator_a: int,
    denominator_a: int,
    numerator_b: int,
    denominator_b: int,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> dict:
    """用精确有理数核对两个分数的加、减、乘或除。"""
    values = (numerator_a, denominator_a, numerator_b, denominator_b)
    if any(abs(value) > 1000000 for value in values):
        return {"error": "整数输入的绝对值须不超过 1000000"}
    try:
        first = Fraction(numerator_a, denominator_a)
        second = Fraction(numerator_b, denominator_b)
        if operation == "add":
            result = first + second
        elif operation == "subtract":
            result = first - second
        elif operation == "multiply":
            result = first * second
        else:
            result = first / second
    except ZeroDivisionError:
        return {"error": "分母和除数不能为零"}
    return {"numerator": result.numerator, "denominator": result.denominator, "exact": str(result)}


@lru_cache(maxsize=1)
def embedding_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    folder = Path(__file__).parent / "model"
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModel.from_pretrained(folder, local_files_only=True, use_safetensors=True)
    return torch, tokenizer, model.eval()


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"


mcp_app = mcp.streamable_http_app()
app = FastAPI(lifespan=mcp_app.router.lifespan_context)


@app.get("/health")
def health():
    return {"status": "ok", "embedding_files": (Path(__file__).parent / "model/model.safetensors").is_file()}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    import base64
    import struct

    texts = [request.input] if isinstance(request.input, str) else request.input
    if request.model != MODEL or not 1 <= len(texts) <= 32 or any(len(text) > 16000 for text in texts):
        raise HTTPException(400, "Unsupported model, batch size or input length")
    torch, tokenizer, model = embedding_model()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state[:, 0]
        vectors = torch.nn.functional.normalize(hidden, p=2, dim=1).tolist()
    data = []
    for index, vector in enumerate(vectors):
        value = vector
        if request.encoding_format == "base64":
            value = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
        data.append({"object": "embedding", "index": index, "embedding": value})
    count = int(inputs["attention_mask"].sum())
    return {"object": "list", "data": data, "model": MODEL, "usage": {"prompt_tokens": count, "total_tokens": count}}


app.mount("/", mcp_app)
