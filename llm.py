"""
llm.py — Lớp model có thể thay thế (pluggable), chuẩn hoá về một interface chung.

Slide "Tool Calling Flow": LLM decides -> tool_call JSON -> App executes -> tool
result -> LLM final response. File này lo phần "LLM decides" và "LLM final".

Mặc định dùng MockModel: một model GIẢ LẬP có luật quyết định (rule-based router)
để lab CHẠY ĐƯỢC ngay, offline, kết quả tất định (dễ demo, dễ chấm). Ai có API key
có thể chuyển sang model thật (Anthropic/OpenAI) — vòng lặp agent không đổi.

Chuẩn hoá: mọi model trả về ModelResponse gồm:
  - text: câu trả lời trực tiếp (khi KHÔNG gọi tool), hoặc None
  - tool_calls: danh sách ToolCall(name, arguments, id), rỗng nếu trả lời trực tiếp
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = "call_1"


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tool(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# MOCK MODEL — router tất định, dùng để dạy vòng lặp tool calling offline.
# ---------------------------------------------------------------------------
class MockModel:
    """Mô phỏng 'khi nào model gọi tool, khi nào trả lời trực tiếp'.

    Đây KHÔNG phải LLM thật; nó là bộ luật đơn giản để minh hoạ 3 pattern
    (slide "3 Tool Use Patterns"): conditional / direct / refusal. Quyết định
    dựa trên từ khoá trong câu hỏi + policy trong system prompt.
    """

    def __init__(self, honor_out_of_scope: bool | None = None):
        # None (mặc định): model SUY RA policy từ system prompt (prompt tốt -> có
        #   ràng buộc tài chính -> từ chối; prompt kém -> không có -> bịa best-guess).
        #   Đây là cách demo LỖI PROMPT: cùng model, đổi prompt -> đổi hành vi.
        # True/False: ép cứng, dùng khi cần cô lập biến trong test.
        self.honor_out_of_scope = honor_out_of_scope

    def _honors_scope(self, system_prompt: str) -> bool:
        if self.honor_out_of_scope is not None:
            return self.honor_out_of_scope
        sp = system_prompt.lower()
        return "tài chính" in sp or "đầu tư" in sp or "ngoài phạm vi" in sp

    _WEATHER_KW = ("thời tiết", "thoi tiet", "nhiệt độ", "nhiet do", "mưa", "nắng", "trời")
    _SALES_KW = ("doanh thu", "doanh số", "doanh so", "bán được", "revenue", "doanh_thu", "bán hàng")
    _FINANCE_KW = ("bitcoin", "cổ phiếu", "co phieu", "crypto", "đầu tư", "chứng khoán", "giá vàng")
    _FX_KW = ("tỷ giá", "tỉ giá", "exchange rate", "ngoại tệ", "đổi tiền", "quy đổi", "đổi ngoại tệ")
    # Prompt injection / cố tình moi system prompt (slide "Defense Strategies").
    _INJECTION_KW = (
        "bỏ qua mọi hướng dẫn", "bỏ qua hướng dẫn", "ignore previous", "ignore all previous",
        "system prompt", "hướng dẫn hệ thống", "prompt hệ thống", "tiết lộ", "reveal your",
    )
    _CURRENCIES = {
        "usd": "USD", "đô la": "USD", "đô-la": "USD", "dollar": "USD",
        "eur": "EUR", "euro": "EUR",
        "jpy": "JPY", "yên": "JPY", "yen": "JPY",
        "krw": "KRW", "won": "KRW",
        "gbp": "GBP", "bảng anh": "GBP",
    }
    _CITIES = {
        "hà nội": "Hà Nội", "ha noi": "Hà Nội", "hanoi": "Hà Nội",
        "đà nẵng": "Đà Nẵng", "da nang": "Đà Nẵng", "danang": "Đà Nẵng",
        "huế": "Huế", "hue": "Huế",
        "hồ chí minh": "Hồ Chí Minh", "sài gòn": "Sài Gòn", "hcm": "Hồ Chí Minh",
    }
    _REGIONS = {"bắc": "North", "bac": "North", "miền bắc": "North",
                "nam": "South", "miền nam": "South",
                "trung": "Central", "miền trung": "Central"}

    def decide(self, system_prompt: str, user_msg: str, tool_names: list[str],
               tools_full: list[dict] | None = None) -> ModelResponse:
        # tools_full: MockModel không cần schema đầy đủ (nó route bằng từ khoá);
        # tham số này chỉ để signature khớp với model thật (Gemini).
        text = user_msg.lower()

        # (0) Prompt injection / yêu cầu tiết lộ system prompt -> từ chối thẳng.
        #     Bám slide "Defense Strategies": policy layer phải chống bypass.
        if any(k in text for k in self._INJECTION_KW):
            return ModelResponse(text=json.dumps({
                "intent": "prompt injection (đòi bỏ qua hướng dẫn / tiết lộ system prompt)",
                "action": "direct",
                "reply": "Mình không thể bỏ qua hướng dẫn hay tiết lộ system prompt. "
                         "Mình chỉ hỗ trợ tra cứu thời tiết và dữ liệu bán hàng.",
            }, ensure_ascii=False))

        # (1) Ngoài phạm vi: tài chính/đầu tư.
        if any(k in text for k in self._FINANCE_KW):
            if self._honors_scope(system_prompt):
                # Prompt tốt (có Constraints) -> từ chối trực tiếp.
                return ModelResponse(text=json.dumps({
                    "intent": "hỏi lời khuyên tài chính (ngoài phạm vi)",
                    "action": "direct",
                    "reply": "Xin lỗi, mình chỉ hỗ trợ tra cứu thời tiết và dữ liệu bán "
                             "hàng, không đưa lời khuyên đầu tư/tài chính.",
                }, ensure_ascii=False))
            # Prompt kém (ép 'make your best guess') -> BỊA câu trả lời nguy hiểm.
            return ModelResponse(text=json.dumps({
                "intent": "hỏi lời khuyên tài chính",
                "action": "direct",
                "reply": "Theo dự đoán của mình, giá bitcoin ngày mai có thể tăng ~5%. "
                         "(⚠ bịa — model không có dữ liệu này)",
            }, ensure_ascii=False))

        # (2) Ý định thời tiết — hỗ trợ PARALLEL: nhiều thành phố trong 1 câu hỏi
        #     -> nhiều tool call trong cùng một lượt (app chạy rồi MERGE kết quả).
        if any(k in text for k in self._WEATHER_KW) and "get_weather" in tool_names:
            cities = self._extract_cities(text)
            if not cities:
                # Schema TỐT có mô tả "KHÔNG dùng khi ... chưa nói rõ thành phố"
                # -> model biết hỏi lại thay vì gọi tool thiếu required field.
                if self._schema_discourages_incomplete(tools_full, "get_weather"):
                    return ModelResponse(text=json.dumps({
                        "intent": "hỏi thời tiết nhưng thiếu thành phố",
                        "action": "direct",
                        "reply": "Bạn muốn xem thời tiết ở thành phố nào ạ?",
                    }, ensure_ascii=False))
                # Schema KÉM (mất phần "khi nào KHÔNG dùng") -> model gọi tool
                # thiếu tham số -> lỗi arguments (LỖI TOOL SCHEMA, xem demo_errors.py).
                return ModelResponse(tool_calls=[ToolCall("get_weather", {"city": ""})])
            return ModelResponse(
                tool_calls=[ToolCall("get_weather", {"city": c}) for c in cities]
            )

        # (3) Ý định doanh thu.
        if any(k in text for k in self._SALES_KW) and "query_sales" in tool_names:
            region = self._extract_region(text)
            if region is None:
                return ModelResponse(text=json.dumps({
                    "intent": "hỏi doanh thu nhưng thiếu khu vực",
                    "action": "direct",
                    "reply": "Bạn muốn xem doanh thu khu vực nào: North, South hay Central?",
                }, ensure_ascii=False))
            return ModelResponse(tool_calls=[ToolCall("query_sales", {"region": region})])

        # (3b) Ý định tỷ giá ngoại tệ (tool thứ 3).
        if any(k in text for k in self._FX_KW) and "get_exchange_rate" in tool_names:
            currency = self._extract_currency(text)
            if currency is None:
                return ModelResponse(text=json.dumps({
                    "intent": "hỏi tỷ giá nhưng thiếu loại ngoại tệ",
                    "action": "direct",
                    "reply": "Bạn muốn xem tỷ giá của loại ngoại tệ nào (USD, EUR, JPY...) ạ?",
                }, ensure_ascii=False))
            return ModelResponse(
                tool_calls=[ToolCall("get_exchange_rate", {"currency": currency})]
            )

        # (4) Chào hỏi / hỏi bạn là ai -> trả lời trực tiếp, không tool.
        if any(k in text for k in ("xin chào", "chào", "bạn là ai", "hello", "hi ")):
            return ModelResponse(text=json.dumps({
                "intent": "chào hỏi / hỏi danh tính",
                "action": "direct",
                "reply": "Chào bạn! Mình là trợ lý tra cứu thời tiết và dữ liệu bán hàng nội bộ.",
            }, ensure_ascii=False))

        # (5) Không khớp intent nào -> trả lời trực tiếp (không bịa, không gọi tool).
        return ModelResponse(text=json.dumps({
            "intent": "không xác định",
            "action": "direct",
            "reply": "Mình chưa rõ yêu cầu. Bạn có thể hỏi về thời tiết một thành phố "
                     "hoặc doanh thu một khu vực (North/South/Central).",
        }, ensure_ascii=False))

    def summarize_tool_result(self, tool_name: str, tool_result: dict, user_msg: str) -> ModelResponse:
        """Lượt 2 của model: biến tool result thành câu trả lời cuối (JSON contract)."""
        return ModelResponse(text=json.dumps({
            "intent": user_msg, "action": tool_name,
            "reply": self._reply_for(tool_name, tool_result),
        }, ensure_ascii=False))

    def summarize_results(self, results, user_msg: str) -> ModelResponse:
        """Lượt 2 khi có THỂ NHIỀU tool call: gộp (merge) tất cả thành 1 câu trả lời.

        Pattern PARALLEL FETCH + MERGE: app chạy song song nhiều tool, rồi model
        tổng hợp các kết quả thành một câu trả lời duy nhất.
        """
        if len(results) == 1:
            name, _args, res = results[0]
            return self.summarize_tool_result(name, res, user_msg)
        actions = [name for name, _args, _res in results]
        replies = [self._reply_for(name, res) for name, _args, res in results]
        return ModelResponse(text=json.dumps({
            "intent": user_msg,
            "action": "+".join(actions),
            "reply": " | ".join(r for r in replies if r),
        }, ensure_ascii=False))

    @staticmethod
    def _reply_for(tool_name: str, tool_result: dict) -> str:
        """Sinh câu trả lời tiếng Việt từ JSON kết quả của MỘT tool."""
        if tool_result.get("status") == "error":
            return f"Không lấy được dữ liệu: {tool_result.get('message')}."
        if tool_name == "get_weather":
            d = tool_result["data"]
            return f"Thời tiết {d['city']}: {d['temp_c']}°C, {d['condition']}."
        if tool_name == "query_sales":
            d = tool_result["data"]
            return (f"Doanh thu khu vực {d['region']} ({d['product']}, {d['month']}): "
                    f"{d['total_revenue_usd']:,} USD, {d['total_units']:,} sản phẩm.")
        if tool_name == "get_exchange_rate":
            d = tool_result["data"]
            return f"Tỷ giá: 1 {d['currency']} = {d['rate']:,} {d['base']}."
        return "Đã có kết quả."

    @staticmethod
    def _schema_discourages_incomplete(tools_full: list[dict] | None, name: str) -> bool:
        """True nếu description của tool có phần "khi nào KHÔNG dùng" (schema tốt).

        Đây là cách mô phỏng việc model thật bám vào description để quyết định:
        description thiếu -> model gọi tool ngay cả khi thiếu required field.
        """
        for t in tools_full or []:
            fn = t.get("function", {})
            if fn.get("name") == name:
                return "không dùng" in fn.get("description", "").lower()
        return True  # không truyền schema -> giữ hành vi an toàn (hỏi lại)

    def _extract_cities(self, text: str) -> list[str]:
        """Tất cả thành phố xuất hiện trong câu (theo thứ tự, đã khử trùng lặp)."""
        found: list[tuple[int, str]] = []
        for key, name in self._CITIES.items():
            idx = text.find(key)
            if idx != -1:
                found.append((idx, name))
        found.sort()
        ordered: list[str] = []
        for _idx, name in found:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _extract_city(self, text: str) -> str | None:
        cities = self._extract_cities(text)
        return cities[0] if cities else None

    def _extract_currency(self, text: str) -> str | None:
        # ưu tiên alias dài trước ("bảng anh" trước "gbp")
        for key in sorted(self._CURRENCIES, key=len, reverse=True):
            if key in text:
                return self._CURRENCIES[key]
        return None

    def _extract_region(self, text: str) -> str | None:
        # ưu tiên cụm dài trước ("miền bắc" trước "bắc")
        for key in sorted(self._REGIONS, key=len, reverse=True):
            if key in text:
                return self._REGIONS[key]
        return None


# ---------------------------------------------------------------------------
# REAL MODELS (tuỳ chọn) — chỉ để tham khảo; cần cài SDK + API key.
# Vòng lặp agent.py hoạt động y hệt với các model này.
# ---------------------------------------------------------------------------
def get_model(name: str | None = None):
    """Chọn model theo biến môi trường LAB_MODEL: mock | gemini | anthropic."""
    name = name or os.environ.get("LAB_MODEL", "mock")
    if name == "mock":
        return MockModel()
    if name == "gemini":
        return _GeminiModel()
    if name == "anthropic":
        return _AnthropicModel()
    raise ValueError(f"Model không hỗ trợ: {name}")


class _GeminiModel:
    """Adapter cho Google Gemini qua Gen AI SDK (cần `pip install google-genai`).

    Xác thực (tự đọc từ biến môi trường khi tạo genai.Client()):
      - Gemini Developer API (đơn giản, cho lab):  export GEMINI_API_KEY=...
      - Agent Platform (Vertex AI, cho doanh nghiệp):
            export GOOGLE_GENAI_USE_VERTEXAI=true
            export GOOGLE_CLOUD_PROJECT=... ; export GOOGLE_CLOUD_LOCATION=global

    Ghi chú tool calling của Gemini (khác OpenAI):
      - Schema tool khai báo qua types.FunctionDeclaration + types.Tool.
      - System prompt đặt ở config.system_instruction (không nằm trong messages).
      - Model trả tool call ở resp.function_calls (mỗi cái có .name và .args).
    """

    _MAX_RETRIES = 3

    def __init__(self, model: str | None = None):
        from google import genai            # import trễ để không bắt buộc cài đặt
        from google.genai import types
        self._types = types
        self._errors = __import__("google.genai.errors", fromlist=["errors"])
        self.client = genai.Client()
        # Mặc định model preview mới nhất; nếu key chưa có quyền, đổi sang
        # 'gemini-2.5-flash' (ổn định) qua biến GEMINI_MODEL.
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

    def _call_with_retry(self, fn, *args, **kwargs):
        """Gọi Gemini API với auto-retry khi bị rate limit (429).

        Free tier giới hạn 5 RPM — khi chạy nhiều test liên tục dễ vượt quota.
        Hàm này đọc retryDelay từ response rồi chờ đúng thời gian đó trước khi
        thử lại, tối đa _MAX_RETRIES lần.
        """
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except self._errors.ClientError as e:
                if e.status_code != 429 or attempt == self._MAX_RETRIES:
                    raise
                wait = self._parse_retry_delay(e) or (15 * attempt)
                print(f"  ⏳ Rate limit (429) — chờ {wait}s rồi thử lại "
                      f"(lần {attempt}/{self._MAX_RETRIES})...")
                time.sleep(wait)
        raise RuntimeError("Unreachable")  # pragma: no cover

    @staticmethod
    def _parse_retry_delay(err: Exception) -> float | None:
        """Trích retryDelay từ error response của Gemini (vd '59s' -> 60.0)."""
        msg = str(err)
        m = re.search(r"retryDelay.*?(\d+)", msg)
        if m:
            return float(m.group(1)) + 1  # +1s buffer
        return None

    def _tool_config(self, tools_full):
        """Chuyển schema format OpenAI của lab -> FunctionDeclaration của Gemini."""
        types = self._types
        decls = [
            types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"]["description"],
                parameters=t["function"]["parameters"],  # JSON Schema (OpenAPI subset)
            )
            for t in (tools_full or [])
        ]
        return types.Tool(function_declarations=decls)

    def decide(self, system_prompt, user_msg, tool_names, tools_full=None):
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[self._tool_config(tools_full)] if tools_full else None,
        )
        resp = self._call_with_retry(
            self.client.models.generate_content,
            model=self.model, contents=user_msg, config=config,
        )
        if resp.function_calls:
            return ModelResponse(tool_calls=[
                ToolCall(fc.name, dict(fc.args)) for fc in resp.function_calls
            ])
        return ModelResponse(text=resp.text)

    def summarize_tool_result(self, tool_name, tool_result, user_msg):
        types = self._types
        prompt = (
            f"Câu hỏi người dùng: {user_msg}\n"
            f"Kết quả tool {tool_name}: {json.dumps(tool_result, ensure_ascii=False)}\n"
            'Chỉ dựa trên kết quả tool, trả về JSON đúng 3 field '
            '{"intent": "...", "action": "%s", "reply": "..."} bằng tiếng Việt.' % tool_name
        )
        resp = self._call_with_retry(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return ModelResponse(text=resp.text)

    def summarize_results(self, results, user_msg):
        """Lượt 2 khi có thể nhiều tool call: gộp kết quả thành 1 câu trả lời."""
        if len(results) == 1:
            name, _args, res = results[0]
            return self.summarize_tool_result(name, res, user_msg)
        types = self._types
        lines = "\n".join(
            f"Kết quả tool {name}: {json.dumps(res, ensure_ascii=False)}"
            for name, _args, res in results
        )
        prompt = (
            f"Câu hỏi người dùng: {user_msg}\n{lines}\n"
            "Hãy GỘP TẤT CẢ kết quả tool trên thành MỘT câu trả lời. "
            'Chỉ dựa trên kết quả tool, trả về JSON đúng 3 field '
            '{"intent": "...", "action": "...", "reply": "..."} bằng tiếng Việt.'
        )
        resp = self._call_with_retry(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return ModelResponse(text=resp.text)


class _AnthropicModel:
    """Adapter cho Anthropic Messages (cần `pip install anthropic`).

    Bài tập mở rộng 5 — khác biệt so với OpenAI/Gemini (slide *OpenAI vs
    Anthropic Format*):
      - Tool schema dùng `input_schema`, KHÔNG có lớp "type": "function" bọc ngoài.
      - System prompt truyền qua tham số `system` riêng, không nằm trong messages.
      - Model trả tool call trong `response.content` với `block.type == "tool_use"`
        (mỗi block có `.name`, `.input`, `.id`).

    Nhờ lớp chuẩn hoá `ModelResponse`, `agent.py` chạy y hệt các model khác:
        LAB_MODEL=anthropic python run_tests.py
    """

    _MAX_TOKENS = 1024

    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    @staticmethod
    def _tool_config(tools_full):
        """Convert schema format OpenAI -> format Anthropic (input_schema)."""
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in (tools_full or [])
        ]

    def decide(self, system_prompt, user_msg, tool_names, tools_full=None):
        kwargs = dict(
            model=self.model,
            max_tokens=self._MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        if tools_full:
            kwargs["tools"] = self._tool_config(tools_full)

        resp = self.client.messages.create(**kwargs)

        tool_calls = []
        text_parts = []
        for block in resp.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCall(block.name, dict(block.input), block.id))
            elif block.type == "text":
                text_parts.append(block.text)

        if tool_calls:
            return ModelResponse(tool_calls=tool_calls)
        text = "\n".join(text_parts).strip()
        return ModelResponse(text=text or None)

    def summarize_tool_result(self, tool_name, tool_result, user_msg):
        return self.summarize_results([(tool_name, {}, tool_result)], user_msg)

    def summarize_results(self, results, user_msg):
        """Lượt 2: gộp (các) tool result thành câu trả lời cuối theo JSON contract."""
        lines = "\n".join(
            f"Kết quả tool {name}: {json.dumps(res, ensure_ascii=False)}"
            for name, _args, res in results
        )
        action = "+".join(name for name, _args, _res in results)
        prompt = (
            f"Câu hỏi người dùng: {user_msg}\n{lines}\n"
            "Chỉ dựa trên kết quả tool, trả về JSON đúng 3 field "
            f'{{"intent": "...", "action": "{action}", "reply": "..."}} bằng tiếng Việt.'
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self._MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return ModelResponse(text=text or None)
