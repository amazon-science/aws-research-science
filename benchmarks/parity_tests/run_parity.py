"""
Parity test suite: openai.gpt-5.4 on Bedrock Mantle vs OpenAI Responses API spec.
Each test makes a single small call and checks that the feature works.
Results saved to results.txt in this directory.

Known Mantle constraints vs OAI SaaS:
  - max_output_tokens minimum is 16
  - previous_response_id: responses not stored/retrievable (store=True default not honoured)
  - Supported tool types: function, mcp, custom, namespace, tool_search
  - NOT supported: web_search, file_search, image_generation, computer_use, shell
"""

import os
import sys
import json
import base64
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from aws_bedrock_token_generator import provide_token
from openai import OpenAI

BASE_URL = "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
MODEL = "openai.gpt-5.4"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.txt")

results = []

S3_IMAGE_URI = "s3://<redacted-bucket>/parity_test/test_image.png"
PLOT_PNG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plot_ttft_otps.png")


def make_client():
    token = provide_token()
    return OpenAI(api_key=token, base_url=BASE_URL)


def record(name, passed, detail=""):
    if passed is None:
        status = "SKIP"
        mark = "⚠️ "
    elif passed:
        status = "PASS"
        mark = "✅"
    else:
        status = "FAIL"
        mark = "❌"
    results.append((name, status, detail))
    print(f"  {mark} [{status}] {name}")
    if detail:
        print(f"       {detail}")


def safe_run(name, fn):
    try:
        fn()
    except Exception as e:
        record(name, False, str(e)[:250])


# ── 1. Basic text generation ──────────────────────────────────────────────────
def test_basic_text():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Reply with exactly: hello world"}],
        max_output_tokens=32,
    )
    text = r.output_text.strip().lower()
    record("Basic text generation", "hello world" in text, repr(text))


# ── 2. Streaming ──────────────────────────────────────────────────────────────
def test_streaming():
    client = make_client()
    chunks = []
    stream = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Count to 3, one number per line."}],
        max_output_tokens=40,
        stream=True,
    )
    for event in stream:
        if getattr(event, "type", None) == "response.output_text.delta":
            chunks.append(event.delta)
    text = "".join(chunks)
    record("Streaming (response.output_text.delta)", len(chunks) > 1 and len(text) > 0,
           f"{len(chunks)} chunks, text={repr(text[:80])}")


# ── 3. Instructions (system prompt) ───────────────────────────────────────────
def test_instructions():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        instructions="Always respond in French, no matter what language the user uses.",
        input=[{"role": "user", "content": "What is the capital of Germany?"}],
        max_output_tokens=40,
    )
    text = r.output_text.lower()
    passed = "berlin" in text
    record("Instructions field (system prompt)", passed, repr(text[:120]))


# ── 4. Multi-role input (developer/user/assistant) ────────────────────────────
def test_multi_role():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 4+4?"},
        ],
        max_output_tokens=16,
    )
    text = r.output_text.strip()
    record("Multi-role input (developer/user/assistant turns)", "8" in text, repr(text))


# ── 5a. Multi-turn via full history (workaround for missing previous_response_id) ─
def test_multiturn_history():
    client = make_client()
    r1 = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "My name is Subshrey. Remember it."}],
        max_output_tokens=40,
    )
    r2 = client.responses.create(
        model=MODEL,
        input=[
            {"role": "user", "content": "My name is Subshrey. Remember it."},
            {"role": "assistant", "content": r1.output_text},
            {"role": "user", "content": "What is my name?"},
        ],
        max_output_tokens=32,
    )
    text = r2.output_text.lower()
    record("Multi-turn via full history (input array)", "subshrey" in text, repr(text))


# ── 5b. Stateful conversation (previous_response_id) ──────────────────────────
def test_stateful_conversation():
    client = make_client()
    r1 = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "My name is Subshrey. Remember it."}],
        max_output_tokens=40,
        store=True,
    )
    try:
        r2 = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "What is my name?"}],
            previous_response_id=r1.id,
            max_output_tokens=32,
        )
        text = r2.output_text.lower()
        record("Stateful conversation (previous_response_id)", "subshrey" in text, repr(text))
    except Exception as e:
        record("Stateful conversation (previous_response_id)", False, str(e)[:200])


# ── 6. Max output tokens ──────────────────────────────────────────────────────
def test_max_output_tokens():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Write a very long essay about the ocean."}],
        max_output_tokens=16,
    )
    out_tokens = r.usage.output_tokens
    record("max_output_tokens enforced (min=16)", out_tokens <= 18, f"output_tokens={out_tokens}")


# ── 7. Temperature ────────────────────────────────────────────────────────────
def test_temperature():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Reply with a single integer between 1 and 10."}],
        max_output_tokens=16,
        temperature=0.0,
    )
    record("Temperature parameter accepted", r.output_text is not None, repr(r.output_text))


# ── 8. Top-p ──────────────────────────────────────────────────────────────────
def test_top_p():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Say yes."}],
        max_output_tokens=16,
        top_p=0.5,
    )
    record("top_p parameter accepted", r.output_text is not None, repr(r.output_text))


# ── 9. Structured output (JSON schema) ───────────────────────────────────────
def test_structured_output():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Extract: 'John is 30 years old and lives in NYC'."}],
        max_output_tokens=80,
        text={
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age":  {"type": "integer"},
                        "city": {"type": "string"},
                    },
                    "required": ["name", "age", "city"],
                    "additionalProperties": False,
                },
            }
        },
    )
    try:
        obj = json.loads(r.output_text)
        passed = obj.get("name") == "John" and obj.get("age") == 30 and "nyc" in obj.get("city", "").lower()
        record("Structured output (json_schema)", passed, repr(obj))
    except Exception as e:
        record("Structured output (json_schema)", False, str(e))


# ── 10. Function calling ──────────────────────────────────────────────────────
def test_function_calling():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "What is the weather in Seattle?"}],
        max_output_tokens=100,
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        }],
    )
    fn_calls = [o for o in r.output if o.type == "function_call"]
    passed = len(fn_calls) > 0 and fn_calls[0].name == "get_weather"
    record("Function calling (single tool)", passed,
           f"{len(fn_calls)} calls, name={fn_calls[0].name if fn_calls else 'none'}")


# ── 11. Parallel function calls ───────────────────────────────────────────────
def test_parallel_function_calls():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "What's the weather in Seattle AND New York?"}],
        max_output_tokens=150,
        parallel_tool_calls=True,
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        }],
    )
    fn_calls = [o for o in r.output if o.type == "function_call"]
    record("Parallel function calls", len(fn_calls) >= 2,
           f"{len(fn_calls)} calls: {[c.name for c in fn_calls]}")


# ── 12. Tool result round-trip (full history, no previous_response_id) ──────────
def test_tool_result_roundtrip():
    client = make_client()
    tool_def = {
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    r1 = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "What is the weather in Boston?"}],
        max_output_tokens=100,
        tools=[tool_def],
    )
    fn_calls = [o for o in r1.output if o.type == "function_call"]
    if not fn_calls:
        record("Tool result round-trip (function_call_output)", False, "No function call in first response")
        return

    # Pass full conversation history + tool result in input array
    r2 = client.responses.create(
        model=MODEL,
        input=[
            {"role": "user", "content": "What is the weather in Boston?"},
            {"type": "function_call", "name": fn_calls[0].name,
             "call_id": fn_calls[0].call_id, "arguments": fn_calls[0].arguments},
            {"type": "function_call_output", "call_id": fn_calls[0].call_id,
             "output": '{"temperature": 55, "condition": "cloudy"}'},
        ],
        max_output_tokens=80,
        tools=[tool_def],
    )
    text = r2.output_text.lower()
    record("Tool result round-trip (function_call_output)", "55" in text or "cloudy" in text, repr(text[:120]))


# ── 13. tool_choice forced ────────────────────────────────────────────────────
def test_tool_choice_forced():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Hello there."}],
        max_output_tokens=80,
        tool_choice={"type": "function", "name": "get_weather"},
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        }],
    )
    fn_calls = [o for o in r.output if o.type == "function_call"]
    record("tool_choice forced", len(fn_calls) > 0, f"{len(fn_calls)} calls")


# ── 14. Image input (HTTPS URL) — expected to fail on Mantle ─────────────────
def test_image_url():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "What color is the sky in this image? One word."},
                {"type": "input_image",
                 "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/1200px-GoldenGateBridge-001.jpg",
                 "detail": "low"},
            ]}],
            max_output_tokens=16,
        )
        text = r.output_text.lower()
        record("Image input (HTTPS URL)", any(w in text for w in ["blue", "gray", "grey", "orange", "sky"]), repr(text))
    except Exception as e:
        record("Image input (HTTPS URL)", False, str(e)[:200])


# ── 15. Image input (data: base64) ───────────────────────────────────────────
def test_image_base64():
    client = make_client()
    # Use the real benchmark plot PNG — confirmed valid PNG file
    with open(PLOT_PNG, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": "This is a benchmark chart. What are the two metrics shown on the Y axes? Answer in a few words."},
            {"type": "input_image",
             "image_url": f"data:image/png;base64,{b64}",
             "detail": "low"},
        ]}],
        max_output_tokens=40,
    )
    record("Image input (data: base64)", len(r.output_text.strip()) > 0, repr(r.output_text[:120]))


# ── 15b. Image input (s3://) ──────────────────────────────────────────────────
def test_image_s3():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "This is a benchmark chart. What are the two metrics shown on the Y axes? Answer in a few words."},
                {"type": "input_image",
                 "image_url": S3_IMAGE_URI,
                 "detail": "low"},
            ]}],
            max_output_tokens=40,
        )
        record("Image input (s3:// URI)", len(r.output_text.strip()) > 0, repr(r.output_text[:120]))
    except Exception as e:
        record("Image input (s3:// URI)", False, str(e)[:200])


# ── 16. Web search tool (not supported on Mantle) ─────────────────────────────
def test_web_search():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "What is today's date?"}],
            max_output_tokens=60,
            tools=[{"type": "web_search_preview"}],
        )
        record("Web search tool", True, repr(r.output_text[:80]))
    except Exception as e:
        err = str(e)
        not_supported = "not supported" in err.lower() or "validation_error" in err.lower()
        record("Web search tool", False, f"NOT SUPPORTED — {err[:150]}")


# ── 17. File search tool (not supported on Mantle) ────────────────────────────
def test_file_search():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Search for AWS docs."}],
            max_output_tokens=40,
            tools=[{"type": "file_search", "vector_store_ids": ["vs_placeholder"]}],
        )
        record("File search tool", True, repr(r.output_text[:80]))
    except Exception as e:
        err = str(e)
        record("File search tool", False, f"NOT SUPPORTED — {err[:150]}")


# ── 18. Tool search (gpt-5.4+ only) ──────────────────────────────────────────
def test_tool_search():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "What is the weather in Denver?"}],
        max_output_tokens=100,
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {"type": "tool_search"},
        ],
    )
    record("Tool search (gpt-5.4+ feature)", len(r.output) > 0,
           f"output types: {[getattr(o,'type','?') for o in r.output]}")


# ── 19. Remote MCP (server_url — not supported, requires connector ARN) ───────
def test_remote_mcp_url():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Roll 2d6."}],
            max_output_tokens=80,
            tools=[{
                "type": "mcp",
                "server_label": "dmcp",
                "server_description": "A Dungeons and Dragons MCP server for dice rolling.",
                "server_url": "https://dmcp-server.deno.dev/sse",
                "require_approval": "never",
            }],
        )
        record("Remote MCP (server_url)", len(r.output_text) > 0,
               f"output_text={repr(r.output_text[:80])}")
    except Exception as e:
        record("Remote MCP (server_url)", False, str(e)[:200])


# ── 19c. Custom tool type ─────────────────────────────────────────────────────
def test_custom_tool():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Calculate 42 * 7 using the calculator tool."}],
        max_output_tokens=60,
        tools=[{
            "type": "custom",
            "name": "calculator",
            "description": "Perform arithmetic calculations. Input is a math expression as a string.",
        }],
    )
    output_types = [getattr(o, "type", "?") for o in r.output]
    has_custom_call = any("custom" in t for t in output_types)
    record("Custom tool type", has_custom_call or len(r.output_text) > 0,
           f"output types: {output_types}, text: {repr(r.output_text[:80])}")


# ── 19d. Namespace tool type ──────────────────────────────────────────────────
def test_namespace_tool():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "What is the weather in Denver?"}],
        max_output_tokens=100,
        tools=[{
            "type": "namespace",
            "name": "weather_tools",
            "description": "Tools for weather information.",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            }],
        }],
    )
    output_types = [getattr(o, "type", "?") for o in r.output]
    record("Namespace tool type", len(r.output) > 0,
           f"output types: {output_types}, text: {repr(r.output_text[:80])}")


# ── 19b. Remote MCP (connector ARN — not testable without provisioned connector)
def test_remote_mcp_arn():
    # We don't have a provisioned MCP connector ARN in this account.
    # Marking as UNTESTED rather than FAIL — this is an infrastructure gap not an API gap.
    record("Remote MCP (connector ARN)", None,
           "UNTESTED — requires a provisioned AWS MCP connector ARN; none available in test account")


# ── 20. Image generation tool (not supported on Mantle) ───────────────────────
def test_image_generation():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Generate a small image of a red circle."}],
            max_output_tokens=50,
            tools=[{"type": "image_generation"}],
        )
        record("Image generation tool", True,
               f"output types: {[getattr(o,'type','?') for o in r.output]}")
    except Exception as e:
        record("Image generation tool", False, f"NOT SUPPORTED — {str(e)[:150]}")


# ── 21. Computer use tool (not supported on Mantle) ───────────────────────────
def test_computer_use():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Click the Submit button."}],
            max_output_tokens=60,
            tools=[{"type": "computer_use_preview", "display_width": 1024, "display_height": 768, "environment": "browser"}],
        )
        record("Computer use tool", True,
               f"output types: {[getattr(o,'type','?') for o in r.output]}")
    except Exception as e:
        record("Computer use tool", False, f"NOT SUPPORTED — {str(e)[:150]}")


# ── 22. Shell tool (not supported on Mantle) ──────────────────────────────────
def test_shell():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Run: echo hello"}],
            max_output_tokens=60,
            tools=[{"type": "shell"}],
        )
        record("Shell tool", True,
               f"output types: {[getattr(o,'type','?') for o in r.output]}")
    except Exception as e:
        record("Shell tool", False, f"NOT SUPPORTED — {str(e)[:150]}")


# ── 23. store=False ───────────────────────────────────────────────────────────
def test_store_false():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Say: stateless"}],
        max_output_tokens=16,
        store=False,
    )
    record("store=False (stateless mode)", "stateless" in r.output_text.lower(), repr(r.output_text))


# ── 24. Usage object ──────────────────────────────────────────────────────────
def test_usage():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Hi."}],
        max_output_tokens=16,
    )
    u = r.usage
    passed = u.input_tokens > 0 and u.output_tokens > 0 and u.total_tokens > 0
    record("Usage object (input/output/total tokens)", passed,
           f"in={u.input_tokens} out={u.output_tokens} total={u.total_tokens}")


# ── 25. Response retrieval by ID ──────────────────────────────────────────────
def test_response_retrieval():
    client = make_client()
    r = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": "Say: retrievable"}],
        max_output_tokens=16,
        store=True,
    )
    retrieved = client.responses.retrieve(r.id)
    record("Response retrieval (GET /responses/{id})", retrieved.id == r.id, f"id={r.id}")


# ── 26. Background mode — accepted, returns immediately ───────────────────────
def test_background_accepted():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Write a haiku about clouds."}],
            background=True,
        )
        has_id = bool(getattr(r, "id", None))
        status = getattr(r, "status", None)
        record("Background mode (background=True accepted)", has_id,
               f"id={r.id}, status={status}")
    except Exception as e:
        record("Background mode (background=True accepted)", False, str(e)[:200])


# ── 27. Background mode — retrieve completed response ─────────────────────────
def test_background_retrieve():
    import time
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Say: background complete"}],
            background=True,
        )
        # Poll up to 30s for completion
        response_id = r.id
        for _ in range(10):
            retrieved = client.responses.retrieve(response_id)
            status = getattr(retrieved, "status", None)
            if status == "completed":
                text = getattr(retrieved, "output_text", "") or ""
                record("Background mode (retrieve until completed)", len(text) > 0,
                       f"status={status}, text={repr(text[:80])}")
                return
            time.sleep(3)
        record("Background mode (retrieve until completed)", False,
               f"Timed out after 30s, last status={status}")
    except Exception as e:
        record("Background mode (retrieve until completed)", False, str(e)[:200])


# ── 28. Background mode — cancel a response ───────────────────────────────────
def test_background_cancel():
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Write a very long 5000 word essay about the history of computing."}],
            background=True,
        )
        cancelled = client.responses.cancel(r.id)
        status = getattr(cancelled, "status", None)
        record("Background mode (cancel response)", status in ("cancelled", "cancelling", "completed"),
               f"status after cancel={status}")
    except Exception as e:
        record("Background mode (cancel response)", False, str(e)[:200])


# ── 29. Background mode — streaming a background response ─────────────────────
def test_background_stream():
    import time
    client = make_client()
    try:
        r = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Count to 5, one number per line."}],
            background=True,
        )
        # Wait briefly then stream
        time.sleep(2)
        chunks = []
        stream = client.responses.retrieve(r.id, stream=True)
        for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                chunks.append(event.delta)
        text = "".join(chunks)
        record("Background mode (stream retrieved response)", len(text) > 0,
               f"{len(chunks)} chunks, text={repr(text[:80])}")
    except Exception as e:
        record("Background mode (stream retrieved response)", False, str(e)[:200])


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ("Basic text generation",                    test_basic_text),
    ("Streaming",                                test_streaming),
    ("Instructions field",                       test_instructions),
    ("Multi-role input",                         test_multi_role),
    ("Multi-turn via full history",              test_multiturn_history),
    ("Stateful conversation (prev_response_id)", test_stateful_conversation),
    ("Max output tokens",                        test_max_output_tokens),
    ("Temperature",                              test_temperature),
    ("Top-p",                                    test_top_p),
    ("Structured output (JSON schema)",          test_structured_output),
    ("Function calling",                         test_function_calling),
    ("Parallel function calls",                  test_parallel_function_calls),
    ("Tool result round-trip",                   test_tool_result_roundtrip),
    ("tool_choice forced",                       test_tool_choice_forced),
    ("Image input (HTTPS URL)",                  test_image_url),
    ("Image input (data: base64)",               test_image_base64),
    ("Image input (s3:// URI)",                  test_image_s3),
    ("Web search tool",                          test_web_search),
    ("File search tool",                         test_file_search),
    ("Tool search",                              test_tool_search),
    ("Remote MCP (server_url)",                  test_remote_mcp_url),
    ("Remote MCP (connector ARN)",               test_remote_mcp_arn),
    ("Custom tool type",                         test_custom_tool),
    ("Namespace tool type",                      test_namespace_tool),
    ("Image generation tool",                    test_image_generation),
    ("Computer use tool",                        test_computer_use),
    ("Shell tool",                               test_shell),
    ("store=False",                              test_store_false),
    ("Usage object",                             test_usage),
    ("Response retrieval by ID",                 test_response_retrieval),
    ("Background mode accepted",                 test_background_accepted),
    ("Background mode retrieve",                 test_background_retrieve),
    ("Background mode cancel",                   test_background_cancel),
    ("Background mode stream",                   test_background_stream),
]


def main():
    started = datetime.now(timezone.utc)
    print(f"\nParity Test Suite: {MODEL} on Bedrock Mantle")
    print(f"Endpoint: {BASE_URL}")
    print(f"Started:  {started.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    for name, fn in TESTS:
        print(f"\n→ {name}")
        safe_run(name, fn)

    ended = datetime.now(timezone.utc)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(results)} passed  |  {failed} failed  |  {skipped} skipped")
    print(f"Ended:   {ended.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    with open(RESULTS_FILE, "w") as f:
        f.write(f"Parity Test Results\n")
        f.write(f"Model:    {MODEL}\n")
        f.write(f"Endpoint: {BASE_URL}\n")
        f.write(f"Started:  {started.isoformat()}\n")
        f.write(f"Ended:    {ended.isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        for name, status, detail in results:
            f.write(f"[{status}] {name}\n")
            if detail:
                f.write(f"       {detail}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"TOTAL: {passed}/{len(results)} passed | {failed} failed | {skipped} skipped\n")

    print(f"Saved:   {RESULTS_FILE}")


if __name__ == "__main__":
    main()
