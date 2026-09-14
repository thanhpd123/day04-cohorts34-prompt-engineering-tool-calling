"""
demo_errors.py — Task 5: Tái hiện & phân loại lỗi thành 3 nhóm.

Câu hỏi mở đầu buổi học: "cùng một agent, đôi khi gọi tool đúng, đôi khi sai —
DO PROMPT hay DO TOOL?". File này trả lời bằng cách CỐ Ý gây ra 3 loại lỗi rồi
phân loại + sửa:

  1. LỖI PROMPT        — system prompt mơ hồ/thiếu ràng buộc -> hành vi sai.
  2. LỖI TOOL SCHEMA   — schema mô tả kém/thiếu enum -> model truyền sai arguments.
  3. LỖI CONTROL FLOW  — vòng lặp agent sai (quên feed tool result) -> câu trả lời hỏng.

Chạy: python demo_errors.py   (mỗi demo in SYMPTOM -> DIAGNOSIS -> FIX)
Chi tiết phân tích nằm trong errors.md.
"""

from __future__ import annotations

import json

from agent import run_agent
from llm import MockModel
from system_prompt import BROKEN_SYSTEM_PROMPT, SYSTEM_PROMPT
from tools import GET_WEATHER_SCHEMA, execute_tool, tool_schemas


def banner(title: str) -> None:
    print(f"\n{'#'*72}\n# {title}\n{'#'*72}")


# ---------------------------------------------------------------------------
# LỖI 1 — PROMPT: prompt mơ hồ, thiếu Constraints -> không từ chối, bịa best-guess.
# ---------------------------------------------------------------------------
def demo_prompt_error() -> None:
    banner("LỖI 1 — PROMPT: thiếu ràng buộc 'ngoài phạm vi' -> agent bịa")
    q = "Bạn nghĩ giá bitcoin ngày mai thế nào?"
    model = MockModel()  # model suy policy TỪ system prompt

    print(">> SYMPTOM (dùng BROKEN_SYSTEM_PROMPT — 'make your best guess'):")
    t_bad = run_agent(q, model=model, system_prompt=BROKEN_SYSTEM_PROMPT, verbose=False)
    print("   FINAL:", t_bad.final_reply)

    print("\n>> DIAGNOSIS: prompt không có ràng buộc NEVER về tài chính/ngoài phạm vi,")
    print("   lại ép 'make your best guess' -> model trả lời câu ngoài phạm vi. Đây là")
    print("   LỖI PROMPT (không phải tool): tool/loop không đổi, chỉ prompt sai.")

    print("\n>> FIX (dùng SYSTEM_PROMPT có Constraints 'NEVER lời khuyên tài chính'):")
    t_ok = run_agent(q, model=model, system_prompt=SYSTEM_PROMPT, verbose=False)
    print("   FINAL:", t_ok.final_reply)


# ---------------------------------------------------------------------------
# LỖI 2 — TOOL SCHEMA: thiếu enum/ví dụ trong schema -> model truyền sai giá trị.
# ---------------------------------------------------------------------------
def demo_tool_schema_error() -> None:
    banner("LỖI 2 — TOOL SCHEMA: thiếu enum -> arguments sai giá trị")

    print(">> SYMPTOM: schema query_sales KHÔNG khai báo enum/ví dụ cho 'region'.")
    print("   Model đoán region theo ngôn ngữ người dùng -> truyền 'miền Bắc':")
    bad_args = {"region": "miền Bắc"}
    res_bad = execute_tool("query_sales", bad_args)
    print("   execute_tool('query_sales',", bad_args, ") ->")
    print("   ", json.dumps(res_bad, ensure_ascii=False))

    print("\n>> DIAGNOSIS: hàm chạy đúng, prompt đúng, nhưng ARGUMENTS sai vì schema")
    print("   không ràng buộc giá trị hợp lệ. Đây là LỖI TOOL SCHEMA.")

    print("\n>> FIX: thêm  \"enum\": [\"North\",\"South\",\"Central\"]  vào schema 'region'")
    print("   (đã có sẵn trong tools.py) để model buộc phải truyền đúng mã:")
    good_args = {"region": "North"}
    res_ok = execute_tool("query_sales", good_args)
    print("   execute_tool('query_sales',", good_args, ") ->")
    print("   ", json.dumps(res_ok, ensure_ascii=False))


# ---------------------------------------------------------------------------
# LỖI 2b — TOOL SCHEMA (biến thể): description mơ hồ -> model gọi tool khi thiếu dữ liệu.
# ---------------------------------------------------------------------------
def demo_tool_description_error() -> None:
    banner("LỖI 2b — TOOL SCHEMA: description mơ hồ -> gọi tool thiếu required field")
    q = "Cho tôi xem thời tiết đi."
    model = MockModel()

    # Schema GOOD: description có phần "KHÔNG dùng khi ... chưa nói rõ thành phố".
    good = tool_schemas()
    names = [t["function"]["name"] for t in good]

    # Schema BAD: chỉ còn "Gets weather." — mất phần "khi nào KHÔNG dùng".
    import copy

    bad = copy.deepcopy(good)
    for t in bad:
        if t["function"]["name"] == "get_weather":
            t["function"]["description"] = "Gets weather."

    print(">> SYMPTOM (description bị làm hỏng, câu hỏi thiếu thành phố):")
    resp_bad = model.decide(SYSTEM_PROMPT, q, names, bad)
    if resp_bad.wants_tool:
        call = resp_bad.tool_calls[0]
        res = execute_tool(call.name, call.arguments)
        print(f"   model GỌI TOOL: {call.name}({call.arguments})")
        print("   execute_tool ->", json.dumps(res, ensure_ascii=False))
    else:
        print("   model trả lời trực tiếp:", resp_bad.text)

    print("\n>> DIAGNOSIS: hàm get_weather vẫn đúng, prompt đúng, nhưng DESCRIPTION mất")
    print("   phần 'KHÔNG dùng khi chưa rõ thành phố' -> model gọi tool thiếu required")
    print("   field. Đây là LỖI TOOL SCHEMA (sửa mô tả schema, không sửa hàm).")

    print("\n>> FIX (khôi phục phần 'KHÔNG dùng ...' trong description):")
    resp_ok = model.decide(SYSTEM_PROMPT, q, names, good)
    if resp_ok.wants_tool:
        print(f"   model GỌI TOOL: {resp_ok.tool_calls[0].name}({resp_ok.tool_calls[0].arguments})")
    else:
        print("   model hỏi lại (direct):", resp_ok.text)


# ---------------------------------------------------------------------------
# LỖI 3 — CONTROL FLOW: vòng lặp quên feed tool result trở lại model.
# ---------------------------------------------------------------------------
def run_agent_broken(user_msg: str, model=None) -> str:
    """Bản agent loop HỎNG: gọi tool nhưng KHÔNG feed kết quả lại cho model."""
    model = model or MockModel()
    resp = model.decide(SYSTEM_PROMPT, user_msg, ["get_weather", "query_sales"])
    if resp.wants_tool:
        call = resp.tool_calls[0]
        _ = execute_tool(call.name, call.arguments)  # BUG: bỏ quên kết quả!
        # Quên bước "summarize_tool_result" -> không có câu trả lời cuối.
        return "(không có câu trả lời — model chưa từng thấy kết quả tool)"
    return resp.text


def demo_control_flow_error() -> None:
    banner("LỖI 3 — CONTROL FLOW: quên feed tool result -> câu trả lời rỗng")
    q = "Thời tiết Hà Nội hôm nay thế nào?"

    print(">> SYMPTOM (run_agent_broken — thiếu bước feed kết quả về model):")
    print("   FINAL:", run_agent_broken(q))

    print("\n>> DIAGNOSIS: tool được gọi đúng, schema đúng, prompt đúng — nhưng VÒNG LẶP")
    print("   agent thiếu bước 4 (tool result -> LLM final). Đây là LỖI CONTROL FLOW.")

    print("\n>> FIX (run_agent chuẩn — có đủ 4 bước của Tool Calling Flow):")
    t_ok = run_agent(q, verbose=False)
    print("   FINAL:", t_ok.final_reply)


if __name__ == "__main__":
    demo_prompt_error()
    demo_tool_schema_error()
    demo_tool_description_error()
    demo_control_flow_error()
    print("\nTóm lại: cùng một câu hỏi có thể hỏng vì PROMPT, vì TOOL SCHEMA, hoặc vì")
    print("CONTROL FLOW. Chẩn đoán đúng NHÓM lỗi mới sửa đúng chỗ. Xem errors.md.")
