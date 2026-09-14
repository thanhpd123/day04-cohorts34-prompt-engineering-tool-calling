"""consistency_check.py — Bài tập mở rộng 3: đo ĐỘ ỔN ĐỊNH (consistency).

Chạy mỗi câu test trong `run_tests.TEST_CASES` N lần rồi tính tỉ lệ PASS.
Mục đích: phát hiện hành vi "lúc đúng lúc sai" của model thật.

    python consistency_check.py                 # MockModel (tất định -> 100%)
    LAB_MODEL=gemini python consistency_check.py --runs 5

Lưu ý free tier Gemini = 5 RPM: script tự `sleep` giữa các lượt gọi để tránh 429.
Ngưỡng tham khảo (slide *Prompt Evaluation Framework*): dưới 90% pass thì cần
iterate prompt/schema.
"""

from __future__ import annotations

import argparse
import os
import time

from agent import run_agent
from llm import MockModel, get_model
from run_tests import TEST_CASES


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo consistency của agent.")
    parser.add_argument("--runs", type=int, default=10, help="số lần chạy mỗi câu")
    parser.add_argument("--sleep", type=float, default=None,
                        help="giây nghỉ giữa các lượt (mặc định 13s với model thật)")
    args = parser.parse_args()

    model = get_model()
    is_mock = isinstance(model, MockModel)
    # Free tier 5 RPM (~12s/lượt). MockModel tất định nên không cần nghỉ.
    sleep_s = args.sleep if args.sleep is not None else (0.0 if is_mock else 13.0)

    print(f"Model: {type(model).__name__} | runs/câu: {args.runs} | sleep: {sleep_s}s")
    print(f"{'='*70}")

    total_ok = total = 0
    for i, (question, expected_tools, _note) in enumerate(TEST_CASES, 1):
        expected = list(expected_tools or [])
        ok_count = 0
        for _ in range(args.runs):
            trace = run_agent(question, model=model, verbose=False)
            if list(trace.tool_calls) == expected:
                ok_count += 1
            if sleep_s:
                time.sleep(sleep_s)

        rate = ok_count / args.runs
        total_ok += ok_count
        total += args.runs
        flag = "✅" if rate >= 0.9 else "⚠️ "
        print(f"  {flag} Test {i}: {ok_count}/{args.runs} PASS ({rate:.0%})  — {question}")

    overall = total_ok / total if total else 0.0
    print(f"{'='*70}")
    print(f"CONSISTENCY TỔNG: {total_ok}/{total} = {overall:.1%}")
    if overall < 0.9:
        print("→ Dưới 90%: cần iterate prompt/schema (xem errors.md).")
    else:
        print("→ Đạt ngưỡng 90%. MockModel tất định nên luôn 100%; model thật cần đo lại.")


if __name__ == "__main__":
    main()
