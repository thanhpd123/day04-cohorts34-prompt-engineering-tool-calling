"""Minh hoạ Zero-shot, One-shot, Few-shot, Chain-of-Thought prompting với Gemini.

So sánh 4 kỹ thuật prompting trên cùng một bài toán: phân loại cảm xúc
(sentiment) của review sản phẩm thành "tích cực", "tiêu cực", hoặc "trung lập".

Cách chạy:
    pip install google-genai          # nếu chưa cài
    export GEMINI_API_KEY=...         # lấy tại https://aistudio.google.com/apikey
    python prompting_examples.py
"""

from __future__ import annotations

import json
import os
import re
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


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _get_client():
    from google import genai
    return genai.Client()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Retry đơn giản cho rate limit (free tier = 5 RPM).
_MAX_RETRIES = 3

def call_gemini(prompt: str) -> str:
    client = _get_client()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            text = resp.text
            if text is None:
                raise RuntimeError("Gemini trả về response rỗng (không có text).")
            return text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < _MAX_RETRIES:
                wait = 15 * attempt
                print(f"  ⏳ Rate limit — chờ {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gọi Gemini thất bại sau {_MAX_RETRIES} lần thử.")


# ---------------------------------------------------------------------------
# Test sentences (dùng chung cho cả 3 kỹ thuật)
# ---------------------------------------------------------------------------

TEST_REVIEWS = [
    "Sản phẩm này dùng tạm được, không có gì đặc biệt.",
    "Giao hàng quá chậm, đợi 2 tuần mới nhận được.",
    "Pin trâu, màn hình đẹp, dùng rất thích!",
    "Mua về chưa mở hộp, chưa biết chất lượng thế nào.",
    "Sản phẩm thiết kế đẹp nhưng đợi 2 tuần mới nhận được hàng.",
]


# ---------------------------------------------------------------------------
# 1. ZERO-SHOT — không cho ví dụ nào
# ---------------------------------------------------------------------------

def zero_shot(review: str) -> str:
    prompt = (
        'Phân loại câu review sau thành ĐÚNG MỘT trong 3 nhãn: '
        '"tích cực", "tiêu cực", hoặc "trung lập".\n'
        "Chỉ trả về nhãn, không giải thích.\n\n"
        f"Review: {review}\n"
        "Nhãn:"
    )
    return call_gemini(prompt)


# ---------------------------------------------------------------------------
# 2. ONE-SHOT — cho 1 ví dụ mẫu
# ---------------------------------------------------------------------------

def one_shot(review: str) -> str:
    prompt = (
        'Phân loại câu review sau thành ĐÚNG MỘT trong 3 nhãn: '
        '"tích cực", "tiêu cực", hoặc "trung lập".\n'
        "Chỉ trả về nhãn, không giải thích.\n\n"
        "Review: Giao hàng nhanh, đóng gói cẩn thận, rất hài lòng!\n"
        "Nhãn: tích cực\n\n"
        f"Review: {review}\n"
        "Nhãn:"
    )
    return call_gemini(prompt)


# ---------------------------------------------------------------------------
# 3. FEW-SHOT — cho 3 ví dụ mẫu, mỗi nhãn 1 ví dụ
# ---------------------------------------------------------------------------

def few_shot(review: str) -> str:
    prompt = (
        'Phân loại câu review sau thành ĐÚNG MỘT trong 3 nhãn: '
        '"tích cực", "tiêu cực", hoặc "trung lập".\n'
        "Chỉ trả về nhãn, không giải thích.\n\n"
        # Ví dụ 1 — tích cực
        "Review: Giao hàng nhanh, đóng gói cẩn thận, rất hài lòng!\n"
        "Nhãn: tích cực\n\n"
        # Ví dụ 2 — tiêu cực
        "Review: Hàng bị lỗi, liên hệ shop không ai trả lời.\n"
        "Nhãn: tiêu cực\n\n"
        # Ví dụ 3 — trung lập
        "Review: Mua về chưa dùng, chưa biết tốt xấu thế nào.\n"
        "Nhãn: trung lập\n\n"
        # Câu cần phân loại
        f"Review: {review}\n"
        "Nhãn:"
    )
    return call_gemini(prompt)


# ---------------------------------------------------------------------------
# 4. CHAIN-OF-THOUGHT (CoT) — yêu cầu model suy luận từng bước trước khi kết luận
# ---------------------------------------------------------------------------
# CoT đặc biệt hiệu quả với câu review MƠ HỒ / MIXED SENTIMENT, ví dụ:
#   "Sản phẩm thiết kế đẹp nhưng đợi 2 tuần mới nhận được hàng."
# Zero-shot có thể trả "tích cực" hoặc "tiêu cực" tuỳ lần chạy.
# CoT buộc model phân tích cả 2 mặt rồi mới đưa nhãn → nhất quán hơn.

def zero_shot_cot(review: str) -> str:
    """Zero-shot CoT: thêm 'Hãy suy nghĩ từng bước' — không cần ví dụ."""
    prompt = (
        'Phân loại câu review sau thành ĐÚNG MỘT trong 3 nhãn: '
        '"tích cực", "tiêu cực", hoặc "trung lập".\n\n'
        "Hãy suy nghĩ từng bước:\n"
        "1. Liệt kê các từ/cụm từ mang cảm xúc trong câu review.\n"
        "2. Đánh giá cảm xúc tổng thể: tích cực nhiều hơn, tiêu cực nhiều hơn, hay cân bằng?\n"
        "3. Đưa ra nhãn cuối cùng.\n\n"
        f"Review: {review}\n\n"
        "Phân tích:"
    )
    return call_gemini(prompt)


def few_shot_cot(review: str) -> str:
    """Few-shot CoT: cho ví dụ kèm lập luận mẫu — model bắt chước cách suy luận."""
    prompt = (
        'Phân loại câu review sau thành ĐÚNG MỘT trong 3 nhãn: '
        '"tích cực", "tiêu cực", hoặc "trung lập".\n'
        "Với mỗi câu, hãy phân tích từng bước rồi đưa ra nhãn.\n\n"
        # --- Ví dụ 1: tích cực rõ ràng ---
        "Review: Giao hàng nhanh, đóng gói cẩn thận, rất hài lòng!\n"
        "Phân tích:\n"
        '- Từ tích cực: "nhanh", "cẩn thận", "rất hài lòng"\n'
        "- Từ tiêu cực: không có\n"
        "- Tổng thể: hoàn toàn tích cực\n"
        "Nhãn: tích cực\n\n"
        # --- Ví dụ 2: tiêu cực rõ ràng ---
        "Review: Hàng bị lỗi, liên hệ shop không ai trả lời.\n"
        "Phân tích:\n"
        '- Từ tích cực: không có\n'
        '- Từ tiêu cực: "bị lỗi", "không ai trả lời"\n'
        "- Tổng thể: hoàn toàn tiêu cực\n"
        "Nhãn: tiêu cực\n\n"
        # --- Ví dụ 3: mixed → trung lập ---
        "Review: Chất lượng tốt nhưng giá hơi cao.\n"
        "Phân tích:\n"
        '- Từ tích cực: "chất lượng tốt"\n'
        '- Từ tiêu cực: "giá hơi cao"\n'
        "- Tổng thể: có cả mặt tốt lẫn mặt chưa tốt, cân bằng\n"
        "Nhãn: trung lập\n\n"
        # --- Câu cần phân loại ---
        f"Review: {review}\n"
        "Phân tích:"
    )
    return call_gemini(prompt)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    techniques = [
        ("ZERO-SHOT",     zero_shot),
        ("ONE-SHOT",      one_shot),
        ("FEW-SHOT",      few_shot),
        ("ZERO-SHOT-COT", zero_shot_cot),
        ("FEW-SHOT-COT",  few_shot_cot),
    ]

    for review in TEST_REVIEWS:
        print(f'\n{"="*70}')
        print(f"Review: \"{review}\"")
        print(f"{'-'*70}")
        for name, fn in techniques:
            result = fn(review)
            # CoT trả lời dài (nhiều dòng) → hiển thị có indent
            if "\n" in result:
                print(f"  {name}:")
                for line in result.splitlines():
                    print(f"    {line}")
            else:
                print(f"  {name:16s} → {result}")

    print(f'\n{"="*70}')
    print("Ghi chú:")
    print("  - Zero-shot:     không ví dụ → nhanh, ít token, format có thể bất nhất.")
    print("  - One-shot:      1 ví dụ → model hiểu format, chưa thấy hết các nhãn.")
    print("  - Few-shot:      3 ví dụ → model thấy cả 3 nhãn, kết quả nhất quán.")
    print("  - Zero-shot CoT: thêm 'suy nghĩ từng bước' → suy luận tốt hơn, không cần ví dụ.")
    print("  - Few-shot CoT:  ví dụ KÈM lập luận → nhất quán nhất cho câu mixed/mơ hồ.")


if __name__ == "__main__":
    main()
