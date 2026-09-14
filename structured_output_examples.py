"""Minh hoạ 4 kỹ thuật Structured Output Prompting với Google Gemini.

Structured output = ép model trả về dữ liệu có cấu trúc cố định (JSON, enum, ...)
thay vì văn xuôi tự do. Rất quan trọng khi output cần parse bằng code phía sau.

4 kỹ thuật từ đơn giản đến mạnh nhất:
  1. Prompt-only JSON       — chỉ dùng lời dặn trong prompt
  2. response_mime_type     — ép Gemini trả JSON qua config API
  3. response_schema        — khai báo JSON Schema, Gemini đảm bảo đúng cấu trúc
  4. Pydantic + response_schema — dùng Pydantic class để sinh schema tự động

Cách chạy:
    pip install google-genai          # nếu chưa cài
    export GEMINI_API_KEY=...
    python structured_output_examples.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Nạp GEMINI_API_KEY từ file .env (nếu có) để chạy ngay, không cần export thủ công.
try:
    from dotenv import load_dotenv

    _ENV_FILE = Path(__file__).with_name(".env")
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
except ImportError:  # python-dotenv là tuỳ chọn
    pass

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_MAX_RETRIES = 3


def _get_client():
    from google import genai
    return genai.Client()


def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < _MAX_RETRIES:
                wait = 15 * attempt
                print(f"  ⏳ Rate limit — chờ {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Test data — cùng 1 bài toán cho cả 4 kỹ thuật:
#   Input:  một review sản phẩm
#   Output: JSON có 3 field: sentiment, confidence, keywords
# ---------------------------------------------------------------------------

TEST_REVIEW = "Pin trâu dùng 2 ngày, nhưng màn hình hơi tối và loa nhỏ."


# ===================================================================
# 1. PROMPT-ONLY JSON — chỉ dùng instruction trong prompt
# ===================================================================
# Đơn giản nhất, hoạt động với mọi model.
# Nhược điểm: model có thể thêm text thừa, sai field name, hoặc
# trả markdown ```json ... ``` thay vì raw JSON.

def prompt_only_json(review: str) -> str:
    prompt = (
        "Phân tích sentiment của review sản phẩm sau.\n"
        "Trả về ĐÚNG JSON hợp lệ, không thêm text hay markdown nào khác.\n"
        "Format:\n"
        '{"sentiment": "tích cực|tiêu cực|trung lập", '
        '"confidence": 0.0-1.0, '
        '"keywords": ["từ khoá 1", "từ khoá 2"]}\n\n'
        f"Review: {review}"
    )
    client = _get_client()
    resp = _call_with_retry(
        client.models.generate_content,
        model=MODEL, contents=prompt,
    )
    return resp.text.strip()


# ===================================================================
# 2. RESPONSE_MIME_TYPE — ép Gemini trả JSON qua config
# ===================================================================
# Gemini API hỗ trợ response_mime_type="application/json".
# Model BẮT BUỘC trả raw JSON (không markdown, không text thừa).
# Prompt vẫn cần mô tả cấu trúc mong muốn.

def mime_type_json(review: str) -> str:
    from google.genai import types

    prompt = (
        "Phân tích sentiment của review sản phẩm sau.\n"
        "Trả về JSON với 3 field:\n"
        '- "sentiment": một trong "tích cực", "tiêu cực", "trung lập"\n'
        '- "confidence": float 0.0-1.0\n'
        '- "keywords": list các từ khoá cảm xúc trong review\n\n'
        f"Review: {review}"
    )
    client = _get_client()
    resp = _call_with_retry(
        client.models.generate_content,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return resp.text.strip()


# ===================================================================
# 3. RESPONSE_SCHEMA — khai báo JSON Schema, Gemini đảm bảo cấu trúc
# ===================================================================
# Mạnh hơn mime_type: bạn khai báo CHÍNH XÁC cấu trúc output bằng
# JSON Schema → Gemini đảm bảo output khớp schema (đúng field, đúng type,
# đúng enum). Không cần mô tả format trong prompt nữa.

def schema_json(review: str) -> str:
    from google.genai import types

    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "sentiment": types.Schema(
                type=types.Type.STRING,
                enum=["tích cực", "tiêu cực", "trung lập"],
                description="Cảm xúc tổng thể của review",
            ),
            "confidence": types.Schema(
                type=types.Type.NUMBER,
                description="Độ tin cậy từ 0.0 đến 1.0",
            ),
            "keywords": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Các từ/cụm từ mang cảm xúc trong review",
            ),
        },
        required=["sentiment", "confidence", "keywords"],
    )

    prompt = f"Phân tích sentiment của review sản phẩm sau:\n\n{review}"

    client = _get_client()
    resp = _call_with_retry(
        client.models.generate_content,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    return resp.text.strip()


# ===================================================================
# 4. PYDANTIC + RESPONSE_SCHEMA — dùng Python class, tự sinh schema
# ===================================================================
# Thay vì viết types.Schema thủ công, dùng Pydantic model → Gemini SDK
# tự chuyển thành JSON Schema. Code Pythonic hơn, dễ maintain, và
# parse output trực tiếp thành Python object.

def pydantic_json(review: str) -> str:
    from google.genai import types
    from enum import Enum
    from pydantic import BaseModel, Field

    class Sentiment(str, Enum):
        POSITIVE = "tích cực"
        NEGATIVE = "tiêu cực"
        NEUTRAL = "trung lập"

    class ReviewAnalysis(BaseModel):
        sentiment: Sentiment = Field(description="Cảm xúc tổng thể")
        confidence: float = Field(ge=0.0, le=1.0, description="Độ tin cậy")
        keywords: list[str] = Field(description="Từ khoá cảm xúc trong review")

    prompt = f"Phân tích sentiment của review sản phẩm sau:\n\n{review}"

    client = _get_client()
    resp = _call_with_retry(
        client.models.generate_content,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReviewAnalysis,
        ),
    )

    parsed = ReviewAnalysis.model_validate_json(resp.text)
    return f"{parsed.model_dump_json(ensure_ascii=False)}  ← parsed OK: sentiment={parsed.sentiment.value}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    techniques = [
        ("1. PROMPT-ONLY",    prompt_only_json, "Chỉ dùng instruction trong prompt"),
        ("2. MIME-TYPE",      mime_type_json,    "response_mime_type='application/json'"),
        ("3. SCHEMA",         schema_json,       "response_schema=types.Schema(...)"),
        ("4. PYDANTIC",       pydantic_json,     "response_schema=PydanticModel"),
    ]

    print(f"Review test: \"{TEST_REVIEW}\"\n")

    for name, fn, desc in techniques:
        print(f"{'='*70}")
        print(f"{name}  —  {desc}")
        print(f"{'-'*70}")
        result = fn(TEST_REVIEW)
        # Pretty-print JSON nếu parse được
        try:
            raw = result.split("  ←")[0] if "  ←" in result else result
            obj = json.loads(raw)
            print(json.dumps(obj, ensure_ascii=False, indent=2))
            if "←" in result:
                print(result.split("  ←")[1].strip())
        except (json.JSONDecodeError, ValueError):
            print(result)
        print()

    print(f"{'='*70}")
    print("So sánh:")
    print("  1. Prompt-only  → đơn giản, nhưng output có thể lẫn text/markdown thừa.")
    print("  2. Mime-type    → ép raw JSON, nhưng không đảm bảo đúng field/type.")
    print("  3. Schema       → đảm bảo đúng cấu trúc, nhưng viết schema thủ công.")
    print("  4. Pydantic     → đảm bảo đúng cấu trúc + parse thành Python object,")
    print("                    code ngắn gọn, dễ maintain nhất.")


if __name__ == "__main__":
    main()
