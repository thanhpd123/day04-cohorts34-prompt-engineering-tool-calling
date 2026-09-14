"""
agent.py — Task 3: Nối tools vào agent loop.

Cài đặt đúng "Tool Calling Flow" (slide 39):
    LLM decides -> tool_call JSON -> App executes tool -> tool result -> LLM final.

Và "3 Tool Use Patterns" (slide 53):
    1. Conditional: agent tự quyết định gọi tool hay trả lời trực tiếp.
    2. Chaining/loop: lặp cho tới khi model không còn muốn gọi tool.
    3. (Parallel: không dùng ở lab này vì các câu test độc lập, 1 tool/câu.)

Điểm quan trọng (slide "Tool calling là bài toán control flow"):
    - Model KHÔNG tự chạy tool. Application nhận tool_call -> execute -> trả kết quả lại.
    - Phải có giới hạn vòng lặp (max_steps) để tránh loop vô hạn.
    - Phải feed tool result trở lại model, nếu quên -> LỖI CONTROL FLOW.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm import MockModel
from system_prompt import SYSTEM_PROMPT
from tools import execute_tool, tool_schemas


@dataclass
class Trace:
    """Nhật ký một lượt chạy agent — để chứng minh khi nào direct, khi nào tool."""
    user_msg: str
    steps: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    final_reply: str = ""

    def log(self, msg: str) -> None:
        self.steps.append(msg)

    @property
    def called_tool(self) -> bool:
        return len(self.tool_calls) > 0


def run_agent(user_msg: str, model=None, system_prompt: str = SYSTEM_PROMPT,
              max_steps: int = 3, verbose: bool = True) -> Trace:
    """Chạy một lượt hội thoại qua agent loop. Trả về Trace để quan sát/đánh giá."""
    model = model or MockModel()
    trace = Trace(user_msg=user_msg)
    schemas = tool_schemas()
    tool_names = [t["function"]["name"] for t in schemas]

    # --- Bước 1: LLM decides (conditional tool use) ---
    # Truyền cả schema đầy đủ để model thật (Gemini) khai báo được tools;
    # MockModel bỏ qua tham số này.
    resp = model.decide(system_prompt, user_msg, tool_names, schemas)

    step = 0
    while resp.wants_tool and step < max_steps:
        step += 1

        # --- Bước 2+3: App executes TẤT CẢ tool call của lượt này (parallel fetch) ---
        results: list[tuple[str, dict, dict]] = []
        for call in resp.tool_calls:
            trace.log(f"[decide] model muốn gọi tool: {call.name}({call.arguments})")
            trace.tool_calls.append(call.name)

            result = execute_tool(call.name, call.arguments)
            trace.log(f"[execute] {call.name} -> {json.dumps(result, ensure_ascii=False)}")
            results.append((call.name, call.arguments, result))

        # --- Bước 4: feed (các) tool result trở lại model -> final response ---
        # (Quên bước này = LỖI CONTROL FLOW, xem demo_errors.py)
        # Nhiều tool call trong 1 lượt -> summarize_results() sẽ MERGE kết quả
        # (pattern parallel fetch + merge; 1 tool call thì hoạt động như cũ).
        resp = model.summarize_results(results, user_msg)

    # --- Trả lời trực tiếp hoặc kết quả cuối ---
    if not trace.called_tool:
        trace.log("[decide] model trả lời TRỰC TIẾP (không gọi tool)")
    trace.final_reply = resp.text or ""

    if verbose:
        _print_trace(trace)
    return trace


def _print_trace(trace: Trace) -> None:
    print(f"\n{'='*70}")
    print(f"USER: {trace.user_msg}")
    print(f"{'-'*70}")
    for s in trace.steps:
        print("  " + s)
    tag = "TOOL" if trace.called_tool else "DIRECT"
    print(f"{'-'*70}")
    print(f"[{tag}] FINAL: {trace.final_reply}")


if __name__ == "__main__":
    run_agent("Thời tiết Hà Nội hôm nay thế nào?")
    run_agent("Doanh thu miền Bắc bao nhiêu?")
    run_agent("Xin chào, bạn là ai?")
