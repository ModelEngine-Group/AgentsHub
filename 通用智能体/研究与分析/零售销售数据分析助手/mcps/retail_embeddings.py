import hashlib
import math
import re
import unicodedata

from starlette.requests import Request
from starlette.responses import JSONResponse


MODEL = "retail-demo-text-384"
DIMENSIONS = 384


def embed(text: str) -> list[float]:
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())
    vector = [0.0] * DIMENSIONS
    for word in words:
        for size in (1, 2):
            for offset in range(len(word) - size + 1):
                feature = word[offset:offset + size].encode("utf-8")
                hashed = int.from_bytes(hashlib.blake2s(feature, digest_size=4).digest(), "big")
                vector[(hashed >> 1) % DIMENSIONS] += (1 if hashed & 1 else -1) * size
    norm = math.sqrt(sum(weight * weight for weight in vector))
    if not norm:
        vector[0] = 1.0
        return vector
    return [weight / norm for weight in vector]


async def embeddings(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        inputs = payload["input"]
    except (ValueError, TypeError, KeyError):
        return JSONResponse({"error": {"message": "Expected JSON embedding input"}}, status_code=400)
    if isinstance(inputs, str):
        inputs = [inputs]
    if (
        payload.get("model") != MODEL
        or not isinstance(inputs, list)
        or not 1 <= len(inputs) <= 32
        or any(not isinstance(item, str) or len(item) > 16000 for item in inputs)
    ):
        return JSONResponse({"error": {"message": "Unsupported model or input"}}, status_code=400)
    return JSONResponse({
        "object": "list",
        "model": MODEL,
        "data": [
            {"index": index, "object": "embedding", "embedding": embed(item)}
            for index, item in enumerate(inputs)
        ],
    })
