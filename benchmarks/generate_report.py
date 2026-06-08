"""
Generate Word document report: Bedrock Mantle gpt-5.4 — Performance + Parity.
Consolidated from our benchmarks + external parity report (55 tests).
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from datetime import datetime

OUTPUT = "/Users/subshrey/Projects/openai_perf_benchmarking/Bedrock_Mantle_GPT54_Report.docx"
PLOT   = "/Users/subshrey/Projects/openai_perf_benchmarking/plot_ttft_otps.png"


def cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_table(doc, headers, rows, col_widths=None, hdr_color="1F4E79", stripe="EBF3FB"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr_row = table.rows[0]
    for i, h in enumerate(headers):
        c = hdr_row.cells[i]
        c.text = h
        run = c.paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)
        cell_bg(c, hdr_color)
    for ri, row in enumerate(rows):
        tr = table.rows[ri + 1]
        for ci, val in enumerate(row):
            c = tr.cells[ci]
            c.text = str(val)
            c.paragraphs[0].runs[0].font.size = Pt(9)
            if ri % 2 == 0:
                cell_bg(c, stripe)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table


def h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def body(doc, text):
    return doc.add_paragraph(text)


def bullet(doc, bold_prefix, rest):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix + ": ").bold = True
    p.add_run(rest)
    return p


def note(doc, text, size=9):
    p = doc.add_paragraph(text)
    for run in p.runs:
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return p


doc = Document()

# ── Title ─────────────────────────────────────────────────────────────────────
t = doc.add_heading("Amazon Bedrock Mantle — openai.gpt-5.4", 0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
s = doc.add_paragraph("Performance Benchmark & API Parity Report")
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
s.runs[0].font.size = Pt(13)
s.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
d = doc.add_paragraph(
    f"Generated: {datetime.now().strftime('%B %d, %Y')}  |  "
    "Region: us-west-2  |  Endpoint: bedrock-mantle.us-west-2.api.aws/openai/v1"
)
d.alignment = WD_ALIGN_PARAGRAPH.CENTER
d.runs[0].font.size = Pt(9)
d.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)
doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PERFORMANCE BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
h(doc, "1. Performance Benchmark")
body(doc,
    "Single-threaded latency benchmarks across 4 input sizes (1K/5K/10K/20K tokens), multiple output configs, "
    "25 runs per configuration. Models: openai.gpt-5.4 and openai.gpt-5.5 on Bedrock Mantle; "
    "gpt-5.4 and gpt-5.5 on OpenAI SaaS direct. All timing in milliseconds."
)
doc.add_paragraph()

h(doc, "1.1  Test Setup", level=2)
add_table(doc, ["Parameter", "Value"], [
    ("Models tested",    "openai.gpt-5.4 (Mantle us-west-2), openai.gpt-5.5 (Mantle us-east-2), gpt-5.4 (OAI SaaS), gpt-5.5 (OAI SaaS)"),
    ("Mantle endpoint",  "https://bedrock-mantle.<region>.api.aws/openai/v1"),
    ("SaaS endpoint",    "https://api.openai.com/v1"),
    ("Auth",             "IAM credentials via aws_bedrock_token_generator"),
    ("Runs per config",  "25 | Single thread, sequential calls"),
    ("Prompt source",    "Manhattan Project (Wikipedia) — real varied text, no repetition"),
    ("1K prompt",        "1,002 tokens | 5K: 4,750 tokens | 10K: 9,984 tokens | 20K: 20,136 tokens"),
], col_widths=[1.8, 4.7])
doc.add_paragraph()

h(doc, "1.2  Mantle gpt-5.4 (us-west-2)", level=2)
add_table(doc,
    ["Input", "Max Out", "TTFT p50", "TTFT p95", "Tok/s p50", "E2E p50", "E2E p95"],
    [
        ("1K",  "100",    "2,628ms", "3,403ms", "45.8",  "4,851ms",  "7,141ms"),
        ("1K",  "500",    "2,545ms", "2,730ms", "73.8",  "9,331ms",  "12,138ms"),
        ("1K",  "1,000",  "2,597ms", "2,884ms", "72.3",  "16,459ms", "19,014ms"),
        ("5K",  "500",    "540ms",   "875ms",   "29.3",  "17,620ms", "—"),
        ("5K",  "1,000",  "380ms",   "754ms",   "41.2",  "24,776ms", "—"),
        ("5K",  "5,000",  "417ms",   "557ms",   "59.4",  "70,753ms", "—"),
        ("10K", "500",    "3,090ms", "4,907ms", "73.0",  "10,019ms", "14,302ms"),
        ("10K", "1,000",  "3,078ms", "4,014ms", "71.3",  "17,108ms", "17,400ms"),
        ("10K", "5,000",  "3,048ms", "4,830ms", "69.9",  "60,710ms", "74,243ms"),
        ("20K", "1,000",  "3,646ms", "5,097ms", "70.3",  "17,918ms", "19,701ms"),
        ("20K", "5,000",  "3,516ms", "5,281ms", "69.4",  "39,973ms", "53,274ms"),
        ("20K", "10,000", "3,478ms", "3,859ms", "69.5",  "40,656ms", "53,473ms"),
    ],
    col_widths=[0.5, 0.7, 0.9, 0.9, 0.8, 0.9, 0.9],
)
doc.add_paragraph()
note(doc, "Note: 5K TTFT anomalously low (380-540ms) compared to other input sizes — under investigation.")
doc.add_paragraph()

h(doc, "1.3  Mantle gpt-5.5 (us-east-2)", level=2)
add_table(doc,
    ["Input", "Max Out", "TTFT p50", "TTFT p95", "Tok/s p50", "E2E p50", "E2E p95"],
    [
        ("1K",  "100",    "1,781ms", "1,781ms",  "56.2",  "1,808ms",  "2,520ms"),
        ("1K",  "500",    "2,670ms", "5,554ms",  "98.4",  "7,787ms",  "8,848ms"),
        ("1K",  "1,000",  "2,678ms", "11,688ms", "84.9",  "14,488ms", "—"),
        ("5K",  "500",    "4,767ms", "8,868ms",  "110.3", "9,003ms",  "—"),
        ("5K",  "1,000",  "4,369ms", "12,360ms", "88.7",  "16,129ms", "—"),
        ("5K",  "5,000",  "5,853ms", "12,198ms", "70.0",  "58,959ms", "—"),
        ("10K", "500",    "5,148ms", "7,094ms",  "141.6", "7,068ms",  "7,801ms"),
        ("10K", "1,000",  "8,151ms", "15,167ms", "157.7", "14,042ms", "15,989ms"),
        ("10K", "5,000",  "8,359ms", "16,568ms", "87.5",  "51,719ms", "73,380ms"),
        ("20K", "1,000",  "5,143ms", "14,636ms", "82.9",  "17,433ms", "—"),
        ("20K", "5,000",  "5,896ms", "13,029ms", "69.0",  "51,957ms", "—"),
        ("20K", "10,000", "5,151ms", "10,495ms", "67.3",  "54,799ms", "—"),
    ],
    col_widths=[0.5, 0.7, 0.9, 0.9, 0.8, 0.9, 0.9],
)
doc.add_paragraph()

h(doc, "1.4  OAI SaaS gpt-5.4 (direct)", level=2)
add_table(doc,
    ["Input", "Max Out", "TTFT p50", "TTFT p95", "Tok/s p50", "E2E p50", "E2E p95"],
    [
        ("1K",  "100",    "632ms",  "1,207ms", "47.0", "2,808ms",  "—"),
        ("1K",  "500",    "656ms",  "2,450ms", "44.4", "12,027ms", "—"),
        ("1K",  "1,000",  "651ms",  "876ms",   "46.3", "22,320ms", "—"),
        ("5K",  "500",    "659ms",  "799ms",   "49.2", "10,846ms", "—"),
        ("5K",  "1,000",  "673ms",  "1,118ms", "47.1", "21,856ms", "—"),
        ("5K",  "5,000",  "709ms",  "954ms",   "48.4", "84,737ms", "—"),
        ("10K", "500",    "738ms",  "1,598ms", "48.2", "11,083ms", "—"),
        ("10K", "1,000",  "710ms",  "1,713ms", "49.3", "21,070ms", "—"),
        ("10K", "5,000",  "825ms",  "1,074ms", "48.7", "79,837ms", "—"),
        ("20K", "1,000",  "850ms",  "3,686ms", "49.4", "21,307ms", "—"),
        ("20K", "5,000",  "916ms",  "1,106ms", "48.6", "58,683ms", "—"),
        ("20K", "10,000", "856ms",  "1,095ms", "46.9", "58,103ms", "—"),
    ],
    col_widths=[0.5, 0.7, 0.9, 0.9, 0.8, 0.9, 0.9],
)
doc.add_paragraph()

h(doc, "1.5  OAI SaaS gpt-5.5 (direct, preview)", level=2)
add_table(doc,
    ["Input", "Max Out", "TTFT p50", "TTFT p95", "Tok/s p50", "E2E p50", "E2E p95"],
    [
        ("1K",  "100",    "3,065ms",  "3,136ms",  "30.0", "3,458ms",  "—"),
        ("1K",  "500",    "4,651ms",  "8,771ms",  "45.5", "15,266ms", "—"),
        ("1K",  "1,000",  "5,065ms",  "10,064ms", "40.9", "28,543ms", "—"),
        ("5K",  "500",    "6,222ms",  "10,370ms", "56.3", "14,460ms", "—"),
        ("5K",  "1,000",  "6,110ms",  "15,077ms", "49.7", "26,642ms", "—"),
        ("5K",  "5,000",  "6,855ms",  "20,405ms", "41.8", "99,826ms", "—"),
        ("10K", "500",    "10,541ms", "12,882ms", "40.9", "13,068ms", "—"),
        ("10K", "1,000",  "13,056ms", "21,258ms", "83.5", "25,430ms", "—"),
        ("10K", "5,000",  "12,460ms", "23,882ms", "47.6", "94,669ms", "—"),
        ("20K", "1,000",  "5,444ms",  "17,714ms", "51.5", "25,175ms", "—"),
        ("20K", "5,000",  "7,291ms",  "18,198ms", "42.5", "88,775ms", "—"),
        ("20K", "10,000", "5,073ms",  "13,256ms", "42.7", "85,950ms", "—"),
    ],
    col_widths=[0.5, 0.7, 0.9, 0.9, 0.8, 0.9, 0.9],
)
doc.add_paragraph()
note(doc, "Note: OAI SaaS gpt-5.5 is in preview. High TTFT (5-13s at 10K input) likely reflects limited/unwarmed capacity, not production performance.")
doc.add_paragraph()

h(doc, "1.6  Key Comparisons (10K input / 1K output, p50)", level=2)
add_table(doc,
    ["Model", "Backend", "TTFT p50", "Tok/s p50", "E2E p50"],
    [
        ("gpt-5.4", "Mantle us-west-2", "3,078ms", "71.3", "17,108ms"),
        ("gpt-5.4", "OAI SaaS",         "710ms",   "49.3", "21,070ms"),
        ("gpt-5.5", "Mantle us-east-2", "8,151ms", "157.7","14,042ms"),
        ("gpt-5.5", "OAI SaaS (preview)","13,056ms","83.5", "25,430ms"),
        ("AA Claim","OAI SaaS gpt-5.4",  "750ms",   "79.1", "—"),
    ],
    col_widths=[1.2, 1.5, 1.2, 1.0, 1.2],
)
doc.add_paragraph()

h(doc, "1.7  Benchmark Chart", level=2)
body(doc,
    "Top panel: TTFT p50 by input size. Bottom panel: output tok/s p50 with min/max band. "
    "OAI SaaS measured values shown as a single point at 1,000 output tokens. "
    "Artificial Analysis published claim shown as a dotted reference line."
)
doc.add_picture(PLOT, width=Inches(6.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

h(doc, "1.8  Key Performance Findings", level=2)
for title, text in [
    ("Mantle gpt-5.4 TTFT scales with input size",
     "p50 TTFT: 1K→2.6s, 10K→3.1s, 20K→3.6s (~500ms per 10K tokens). "
     "OAI SaaS gpt-5.4 is much faster at 0.6-0.9s p50 regardless of input size."),
    ("gpt-5.5 Mantle throughput is 2× gpt-5.4 at 10K input",
     "Mantle gpt-5.5 delivers 141-157 tok/s p50 at 10K input vs 70-73 for gpt-5.4. "
     "This likely reflects gpt-5.5 using parallel decoding or speculative execution."),
    ("gpt-5.5 TTFT is higher than gpt-5.4 on Mantle",
     "gpt-5.5 TTFT p50 at 10K: 5-8s vs gpt-5.4's 3s. Likely includes reasoning/thinking time. "
     "Despite higher TTFT, E2E is competitive due to faster generation."),
    ("OAI SaaS gpt-5.4 has lowest TTFT (0.6-0.9s)",
     "Consistently fastest TTFT across all input sizes. "
     "This is the baseline that Mantle should target for production-ready latency."),
    ("OAI SaaS gpt-5.5 is in preview — not representative",
     "gpt-5.5 SaaS TTFT is 5-13s at 10K input, far slower than gpt-5.4 SaaS (0.7s). "
     "Mantle gpt-5.5 actually outperforms SaaS gpt-5.5 on TTFT — likely unwarmed capacity on SaaS side."),
    ("Mantle gpt-5.4 throughput competitive with OAI SaaS",
     "gpt-5.4: Mantle 69-73 tok/s vs SaaS 47-49 tok/s. Mantle wins on throughput despite TTFT gap."),
    ("5K input TTFT anomaly on Mantle gpt-5.4",
     "380-540ms TTFT at 5K input is unexpectedly fast (lower than 1K input at 2.5s). "
     "Requires investigation — may be a caching or routing artifact."),
]:
    bullet(doc, title, text)
doc.add_paragraph()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — API PARITY
# ═══════════════════════════════════════════════════════════════════════════════
h(doc, "2. API Feature Parity")
body(doc,
    "Consolidated from two independent test runs: our own 30-test suite against Bedrock Mantle, "
    "and an external 55-test side-by-side comparison of OpenAI 1P (api.openai.com) vs Bedrock Mantle 3P. "
    "Combined coverage: 64 unique tests. Overall parity score from external report: 84% (46/55); "
    "extended to 48/57 including our 3 additional tool type tests. "
    "Our additional tests cover tool types (custom, namespace, tool_search, s3 vision, computer use, shell, "
    "response retrieval) not in the external suite. "
    "Sections 2.3–2.5 use the external report as authoritative source for 1P vs 3P comparisons; "
    "Section 2.5 contains Mantle-specific findings from our suite."
)
doc.add_paragraph()

h(doc, "2.1  Test Setup", level=2)
add_table(doc, ["Parameter", "Value"], [
    ("Model",              "openai.gpt-5.4"),
    ("Mantle endpoint",    "https://bedrock-mantle.us-west-2.api.aws/openai/v1"),
    ("OAI SaaS endpoint",  "https://api.openai.com/v1"),
    ("Auth (Mantle)",      "IAM credentials via aws_bedrock_token_generator"),
    ("Auth (OAI SaaS)",    "Personal API key"),
    ("Per test",           "1 API call, small token budget"),
    ("External suite",     "55 tests, side-by-side 1P vs 3P"),
    ("Our suite",          "30 tests, Mantle only; adds tool type and s3 coverage"),
], col_widths=[1.8, 4.7])
doc.add_paragraph()

h(doc, "2.2  Full Parity — Both Pass (48 tests)", level=2)
body(doc, "All features below work correctly on both OpenAI SaaS and Bedrock Mantle. "
     "Tests 1–45 from external 55-test report; tests 46–48 from our additional suite.")
doc.add_paragraph()

add_table(doc,
    ["#", "Feature", "Category", "Notes"],
    [
        ("1",  "Basic text generation",            "Core API",       ""),
        ("2",  "System instructions",              "Core API",       ""),
        ("3",  "max_output_tokens",                "Core API",       "Mantle minimum is 16; OAI allows 1"),
        ("4",  "Response schema (id, status, ...)", "Core API",      "Object structure matches spec"),
        ("5",  "SSE streaming events",             "Streaming",      "response.output_text.delta delivered correctly"),
        ("6",  "Streaming TTFT",                   "Streaming",      "Time to first token measurable via stream"),
        ("7",  "Single function call",             "Function calling",""),
        ("8",  "tool_choice=required",             "Function calling",""),
        ("9",  "tool_choice=none",                 "Function calling",""),
        ("10", "tool_choice specific function",    "Function calling",""),
        ("11", "parallel_tool_calls=true",         "Function calling",""),
        ("12", "parallel_tool_calls=false",        "Function calling",""),
        ("13", "max_tool_calls parameter",         "Function calling",""),
        ("14", "Tool result round-trip",           "Function calling","function_call_output in input array"),
        ("15", "JSON object mode",                 "Structured output",""),
        ("16", "Strict JSON schema",               "Structured output","text.format json_schema strict=True"),
        ("17", "reasoning.effort=none",            "Reasoning",      ""),
        ("18", "reasoning.effort=low",             "Reasoning",      ""),
        ("19", "reasoning.effort=medium",          "Reasoning",      ""),
        ("20", "reasoning.effort=high",            "Reasoning",      ""),
        ("21", "reasoning.effort=xhigh",           "Reasoning",      ""),
        ("22", "reasoning.summary=auto",           "Reasoning",      ""),
        ("23", "reasoning.summary=concise",        "Reasoning",      ""),
        ("24", "reasoning.summary=detailed",       "Reasoning",      ""),
        ("25", "Encrypted reasoning content",      "Reasoning",      "reasoning.encrypted_content supported"),
        ("26", "Image base64 input",               "Vision",         "data: scheme works; HTTPS URLs rejected on both"),
        ("27", "Image detail parameter",           "Vision",         "detail=low/high/auto accepted"),
        ("28", "Multi-turn via full history",      "Multi-turn",     "input array conversation history"),
        ("29", "Stateless continuation",           "Multi-turn",     ""),
        ("30", "temperature=0 (determinism)",      "Parameters",     ""),
        ("31", "temperature=1.5 (high creativity)","Parameters",     ""),
        ("32", "top_p",                            "Parameters",     ""),
        ("33", "truncation=auto",                  "Context mgmt",   ""),
        ("34", "store=True",                       "Context mgmt",   ""),
        ("35", "store=False",                      "Context mgmt",   ""),
        ("36", "background=True",                  "Background",     "Async processing mode"),
        ("37", "Usage token reporting",            "Usage",          "input/output/total tokens present"),
        ("38", "metadata parameter",               "Metadata",       "key-value pairs accepted"),
        ("39", "service_tier=auto",                "Service tier",   ""),
        ("40", "service_tier=default",             "Service tier",   ""),
        ("41", "text.verbosity=low",               "Verbosity",      ""),
        ("42", "text.verbosity=high",              "Verbosity",      ""),
        ("43", "user parameter",                   "User tracking",  ""),
        ("44", "Invalid model error handling",     "Error handling", "Correct error code and schema returned"),
        ("45", "Error schema format",              "Error handling", "Matches OpenAI error object spec"),
        ("46", "Custom tool type",                 "Tool types",     "type=custom; our test only"),
        ("47", "Namespace tool type",              "Tool types",     "type=namespace; description field required; our test only"),
        ("48", "Tool search",                      "Tool types",     "type=tool_search; gpt-5.4+ only; our test only"),
    ],
    col_widths=[0.3, 2.2, 1.3, 2.7],
    hdr_color="1F6B3A",
)
doc.add_paragraph()

h(doc, "2.3  Bedrock Gaps — Works on OAI SaaS Only (5 tests)", level=2)
add_table(doc,
    ["#", "Feature", "Category", "OAI SaaS", "Bedrock Mantle", "Request ID"],
    [
        ("1", "web_search_preview",          "Web search",    "✅ Pass", "❌ 400 — tool type not supported",
         "req_cpetjj5limghmgygjblptetoxqqantpucacyy7eh3yh2yxdnnama"),
        ("2", "web_search + user_location",  "Web search",    "✅ Pass", "❌ 400 — tool type not supported",
         "req_ozmv57kb2pqv3r4u5hx5eqlsunywvjitlr5m2xnaippud4oiee4a"),
        ("3", "previous_response_id",        "Stateful",      "✅ Pass", "❌ 404 — responses not stored",
         "req_6rq2t26ugo77irk6myg6zrbmnnnpzcdow7xppbk5cnadl5g6xuvq"),
        ("4", "logprobs / include",          "Include",       "✅ Pass", "❌ 400 — unsupported_value; only reasoning.encrypted_content allowed",
         "req_y3exzndu4d33ejfzzyltn3htatejb63cuqtsrudltr4xhv5tehda"),
        ("5", "image_generation tool",       "Hosted tools",  "✅ Pass", "❌ 400 — tool type not supported",
         "req_is756ibh5ovnmg3jjlp346tzk5few5mqlyfmqg2vbg6mrwazr2da"),
    ],
    col_widths=[0.3, 1.6, 1.0, 0.7, 2.0, 2.0],
    hdr_color="8B4000",
)
doc.add_paragraph()

h(doc, "2.4  Both Platforms Fail (4 tests)", level=2)
body(doc, "These are not Mantle-specific gaps — neither platform supports them.")
doc.add_paragraph()
add_table(doc,
    ["#", "Feature", "Category", "1P Error", "3P Error"],
    [
        ("1", "Image URL (HTTPS)",    "Vision",       "400 — not supported",  "400 — unsupported scheme"),
        ("2", "file_search",          "Hosted tools", "404 — no vector store", "400 — tool type not supported"),
        ("3", "code_interpreter",     "Hosted tools", "400 — not supported",  "400 — tool type not supported"),
        ("4", "MCP remote connector", "MCP",          "424 — connection error","400 — use connector ARN"),
    ],
    col_widths=[0.3, 1.5, 1.0, 2.2, 2.2],
    hdr_color="555555",
)
doc.add_paragraph()

h(doc, "2.5  Additional Failures (Mantle only, our tests)", level=2)
body(doc, "These were found in our test suite and are not covered by the external report.")
doc.add_paragraph()
add_table(doc,
    ["#", "Feature", "Error", "Notes"],
    [
        ("1", "Response retrieval GET /responses/{id}", "500 internal_server_error",
         "Route exists but broken server-side. Request ID: req_ro5mc7vnrzjc6cwkevke4krgekhjsnfpjgymad37sh4x4ltvfs7q"),
        ("2", "Image input (data: base64) — large image", "500 internal_server_error",
         "Small JPEG works (confirmed). Large PNG (155KB) triggers server error. Size/format sensitivity."),
        ("3", "Image input (s3:// URI)", "500 internal_server_error",
         "Scheme accepted by URL validator. Real PNG uploaded to same-account S3 bucket still fails server-side."),
        ("4", "Remote MCP server_url", "400 — use connector ARN",
         "MCP type accepted but arbitrary server URLs rejected. Requires AWS connector ARN."),
        ("5", "computer_use_preview", "400 — tool type not supported", ""),
        ("6", "shell", "400 — tool type not supported", ""),
    ],
    col_widths=[0.3, 2.0, 1.5, 2.7],
    hdr_color="8B0000",
)
doc.add_paragraph()

h(doc, "2.6  Key Parity Findings", level=2)
for title, text in [
    ("Overall parity: 84% (46/55) from external report",
     "48 features have full parity, 5 are Bedrock gaps, 4 fail on both platforms. "
     "Our additional tests (custom tool, namespace, tool_search) extend the external report's 44 passing features to 48."),
    ("Reasoning: full parity across all effort levels",
     "All 5 effort levels (none/low/medium/high/xhigh) and all 3 summary modes work. "
     "Encrypted reasoning content supported. This is a significant capability match."),
    ("Function calling: full parity",
     "All tool_choice modes (auto/required/none/specific), parallel calls, max_tool_calls, "
     "and tool result round-trips work. All 5 Mantle tool types confirmed: function, custom, namespace, tool_search, mcp (ARN)."),
    ("Vision: partial — small images work, large images fail",
     "data: base64 works for small JPEGs (confirmed). HTTPS URLs rejected on both platforms. "
     "Large images (>100KB PNG) trigger 500 errors. s3:// scheme fails server-side. "
     "External report confirms base64 works — our 500 was caused by image size."),
    ("Stateful features: not implemented on Mantle",
     "previous_response_id returns 404 — responses not persisted. "
     "Response retrieval (GET /responses/{id}) returns 500. "
     "If stateful features were fixed, parity would rise to ~87% (48/55). "
     "Workaround: pass full conversation history in input array (confirmed working)."),
    ("Logprobs: Mantle gap",
     "include=[\"message.output_text.logprobs\"] returns 400. "
     "Only supported include value is reasoning.encrypted_content."),
    ("Hosted tools gap: web_search and image_generation",
     "web_search_preview and image_generation are OAI-only. "
     "file_search and code_interpreter fail on both platforms."),
    ("Background processing: full parity",
     "background=True for async processing works on Mantle."),
    ("max_output_tokens minimum difference",
     "Mantle enforces minimum of 16 tokens. OAI SaaS allows down to 1."),
]:
    bullet(doc, title, text)
doc.add_paragraph()

h(doc, "2.7  Parity Summary", level=2)
add_table(doc,
    ["Feature Area", "Parity", "Notes"],
    [
        ("Text generation",               "✅ Full",    "Basic, instructions, multi-role, error handling"),
        ("Streaming (SSE)",               "✅ Full",    "Events + TTFT both confirmed"),
        ("Reasoning",                     "✅ Full",    "All effort levels, summary modes, encrypted content"),
        ("Structured output",             "✅ Full",    "JSON schema (strict) + JSON object mode"),
        ("Function calling",              "✅ Full",    "All tool_choice modes, parallel, round-trip, max_tool_calls"),
        ("Tool types",                    "✅ Full",    "function, custom, namespace, tool_search; mcp needs ARN"),
        ("Parameters",                    "✅ Full",    "temperature, top_p, truncation, store, background, metadata"),
        ("Service tier / verbosity",      "✅ Full",    "service_tier=auto/default, text.verbosity=low/high"),
        ("User / usage / error handling", "✅ Full",    "user param, usage tokens, error schema"),
        ("Vision (small base64 JPEG)",    "✅ Full",    "Works for small images; large images trigger 500"),
        ("Vision (HTTPS URL)",            "✗ Both fail","Rejected on both OAI SaaS and Mantle"),
        ("Vision (s3:// URI)",            "❌ Mantle",  "500 server error on Mantle; not tested on OAI SaaS"),
        ("Vision (large images)",         "❌ Mantle",  "500 on Mantle; size/format sensitivity"),
        ("Stateful (previous_response_id)","❌ Mantle", "404 — responses not persisted on Mantle"),
        ("Response retrieval GET",        "❌ Mantle",  "500 server error"),
        ("Logprobs",                      "❌ Mantle",  "400 — not supported; only reasoning.encrypted_content allowed"),
        ("Web search",                    "❌ Mantle",  "Tool type not available"),
        ("Image generation tool",         "❌ Mantle",  "Tool type not available"),
        ("File search",                   "✗ Both fail","Fails on both platforms"),
        ("Code interpreter",              "✗ Both fail","Fails on both platforms"),
        ("MCP (connector ARN)",           "⚠️ Untested","No ARN provisioned; type accepted"),
        ("MCP (server_url)",              "❌ Mantle",  "Arbitrary URLs rejected; ARN required"),
        ("Computer use",                  "❌ Mantle",  "Tool type not available"),
        ("Shell",                         "❌ Mantle",  "Tool type not available"),
    ],
    col_widths=[2.2, 1.0, 3.3],
)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QUALITY BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
h(doc, "3. Quality Benchmarks")
body(doc,
    "Intelligence and capability benchmarks run against openai.gpt-5.4 on Bedrock Mantle and, "
    "where possible, OAI SaaS direct for comparison. Methodology follows Artificial Analysis "
    "where applicable: temperature=0.6, zero-shot prompted, pass@1 scoring. "
    "Note: HLE OAI SaaS run pending (API quota). AIME SaaS completed."
)
doc.add_paragraph()

h(doc, "3.1  Test Setup", level=2)
add_table(doc, ["Parameter", "Value"], [
    ("Model (Mantle)",   "openai.gpt-5.4"),
    ("Model (SaaS)",     "gpt-5.4"),
    ("Temperature",      "0.6 (reasoning model, per AA methodology)"),
    ("Prompting",        "Zero-shot instruction prompted, no examples"),
    ("Scoring",          "Pass@1; failed/blocked calls excluded from answerable accuracy"),
    ("Error handling",   "Full error capture: request_id, status_code, error_code, headers"),
    ("LLM judge",        "us.anthropic.claude-haiku-4-5-20251001-v1:0 via Bedrock (HLE only)"),
], col_widths=[2.0, 4.5])
doc.add_paragraph()

h(doc, "3.2  Summary Results", level=2)
add_table(doc,
    ["Eval", "Mantle", "OAI SaaS", "Published (AA)", "Notes"],
    [
        ("GPQA Diamond",              "72.5%",              "76.8%",              "92.8%", "5 repeats, 198 Qs"),
        ("AIME 2024",                 "83.3% (ex-filter)",  "85.4% (ex-filter)",  "N/A",   "14 Qs; 14.3% vs 2.9% filter blocks"),
        ("HLE (text-only)",           "10.2% (LLM judge)",  "⏳ pending",          "39.8%", "2,158 Qs; exact=5.0%; judge=Haiku"),
        ("─────",                     "─────",              "─────",              "─────", "─────"),
        ("Logic Reasoning",           "49.0%",              "47.1%",              "—",     "Independent internal benchmark"),
        ("Hallucination Detection",   "95.2%",              "95.2%",              "—",     "Independent internal benchmark"),
        ("Information Retrieval",     "1.7%",               "0.0%",               "—",     "Independent internal; Mantle context limit"),
        ("Tool Calling (Hard)",       "48.3% exact",        "46.7% exact",        "—",     "Independent internal benchmark"),
        ("Exam (Hard)",               "3.0%",               "4.0%",               "—",     "Independent internal; LLM-judged"),
    ],
    col_widths=[1.7, 1.3, 1.0, 1.0, 2.5],
    hdr_color="1F4E79",
)
doc.add_paragraph()

h(doc, "3.3  GPQA Diamond", level=2)
body(doc, "198 graduate-level science questions (4-option MCQ), 5 repeats, randomized answer order. "
     "Dataset: Idavidrein/gpqa (gpqa_diamond). Published GPT-5.4 score: 92.8% (Artificial Analysis).")
doc.add_paragraph()

add_table(doc,
    ["Domain", "Mantle", "OAI SaaS", "Questions"],
    [
        ("Physics",   "88.1%", "90.9%", "86"),
        ("Biology",   "69.5%", "72.6%", "19"),
        ("Chemistry", "58.7%", "64.5%", "93"),
        ("Overall",   "72.5%", "76.8%", "198"),
    ],
    col_widths=[1.5, 1.2, 1.2, 1.0],
    hdr_color="1F6B3A",
)
doc.add_paragraph()

for title, text in [
    ("4.3-point Mantle vs SaaS gap",
     "Consistent across all three domains. Physics is strong on both (88–91%). "
     "Chemistry is the weak domain on both platforms (58–64%)."),
    ("Both 16–20 points below published AA score",
     "AA tests from GCP us-central1-a with 8 runs/day rolling p50 median. "
     "Geographic proximity and methodology differences likely explain part of the gap. "
     "Whether Mantle routing also contributes requires further testing."),
    ("No content filter issues",
     "All 990 GPQA calls completed without content filter blocks on either platform."),
]:
    bullet(doc, title, text)
doc.add_paragraph()

h(doc, "3.4  AIME 2024", level=2)
body(doc, "14 competition math problems (AIME II 2024 only — AIME I absent from public dataset). "
     "5 repeats, exact integer match scoring. Both Mantle and OAI SaaS runs completed.")
doc.add_paragraph()

add_table(doc,
    ["Metric", "Mantle", "OAI SaaS"],
    [
        ("Raw accuracy",                    "71.4%  (50/70)", "82.9%  (58/70)"),
        ("Answerable accuracy (ex-filter)", "83.3%  (50/60)", "85.4%  (58/68)"),
        ("Content filter blocks",           "10/70  (14.3%)", "2/70   (2.9%)"),
        ("Q5",  "0/5 (pred 185, correct 80)",    "1/5 correct"),
        ("Q7",  "0/5 — all blocked",              "3/5 — 2 blocked, 3 correct"),
        ("Q8",  "5/5 perfect",                    "4/5 — 1 wrong"),
        ("Q14", "0/5 (pred 350-675, correct 315)", "0/5 genuine miss"),
    ],
    col_widths=[2.2, 2.1, 2.2],
)
doc.add_paragraph()

p = doc.add_paragraph()
p.add_run("Content filter false positives: ").bold = True
p.add_run(
    "Q7 blocked 5/5 on Mantle vs 2/5 on SaaS — Mantle is more aggressive. "
    "All blocked problems are standard competition math with no harmful content. "
    "Filter appears intermittent — could not reproduce in later testing (self-resolved). "
    "Timestamp: 2026-05-28 02:19-02:36 UTC, account <redacted>, us-west-2. "
    "SaaS request IDs: req_004ef290906e4a328be88dc064c843dc, req_ea4a4c3fcd6547c187075ee81af440c5."
)
doc.add_paragraph()

for title, text in [
    ("Raw accuracy gap (71.4% vs 82.9%) almost entirely explained by filter rate",
     "Answerable accuracy is nearly identical: 83.3% (Mantle) vs 85.4% (SaaS). "
     "When both platforms answer the question, they perform equivalently."),
    ("Q7 filtered on both platforms — not purely a Mantle issue",
     "Mantle blocks it more aggressively (5/5 vs 2/5 on SaaS). "
     "The problem involves divisibility by 7 and contains no harmful content."),
    ("Q14 is a genuine model miss on both platforms",
     "Consistently wrong across all 5 repeats on both Mantle and SaaS — knowledge gap, not infrastructure."),
    ("Dataset limitation",
     "Only 14/30 AIME 2024 problems available in the public dataset (AIME I 2024 missing)."),
]:
    bullet(doc, title, text)
doc.add_paragraph()

h(doc, "3.5  HLE — Humanity's Last Exam (text-only)", level=2)
body(doc, "2,158 text-only questions from cais/hle test split (342 image questions excluded). "
     "1 repeat. Exact string match + LLM re-scoring via Claude Haiku 4.5 on Bedrock. "
     "Published GPT-5.4 score: 39.8% (no tools, AA methodology with GPT-4o judge).")
doc.add_paragraph()

add_table(doc,
    ["Scoring Method", "Accuracy", "Notes"],
    [
        ("Exact string match",      "5.0%  (107/2158)",  "Too strict — misses equivalent formats"),
        ("LLM judge (Haiku 4.5)",   "10.2% (220/2158)",  "Catches math notation variants"),
        ("LLM judge (answerable)",  "10.6% (220/2081)",  "Excludes 77 unanswered questions"),
        ("Published GPT-5.4 (AA)",  "39.8%",             "GPT-4o judge, full 2500 Qs"),
    ],
    col_widths=[2.2, 1.5, 2.8],
)
doc.add_paragraph()

for title, text in [
    ("LLM judge doubled the score",
     "Exact string match severely underestimates accuracy — 5% → 10.2% after LLM judging. "
     "Many answers are equivalent but expressed differently (LaTeX, decimals, fractions)."),
    ("30-point gap vs published still unexplained",
     "Haiku may be less lenient than GPT-4o judge. Text-only subset may be harder than the full dataset. "
     "Genuine Mantle vs SaaS performance gap possible. SaaS comparison needed."),
    ("3.6% unanswered rate",
     "77 questions returned no extractable answer — model didn't follow Answer: format or response truncated."),
    ("Image questions excluded",
     "342 of 2,500 HLE questions require vision input, which is non-functional on Mantle."),
]:
    bullet(doc, title, text)
doc.add_paragraph()

h(doc, "3.6  Independent Internal Benchmarks", level=2)
body(doc,
    "Results from a set of internal benchmarks. Benchmark names and scores are reported; "
    "details of benchmark design and datasets are not disclosed."
)
doc.add_paragraph()

add_table(doc,
    ["Benchmark", "Mantle", "OAI SaaS", "Notes"],
    [
        ("Logic Reasoning",        "49.0%", "47.1%", "51 tests"),
        ("Hallucination Detection","95.2%", "95.2%", "21 tests"),
        ("Information Retrieval",  "1.7%",  "0.0%",  "116/129 tests; Mantle context limit applies"),
        ("Tool Calling (Hard)",    "48.3% exact / 0.729 mean", "46.7% exact / 0.717 mean", "120 tests"),
        ("Exam (Hard)",            "3.0%",  "4.0%",  "100 tests; LLM-judged"),
    ],
    col_widths=[1.8, 1.8, 1.8, 2.1],
    hdr_color="1F4E79",
)
doc.add_paragraph()

for title, text in [
    ("Near-identical performance on Logic Reasoning, Hallucination, Tool Calling",
     "Mantle and OAI SaaS score within 1-2 points on these benchmarks — no meaningful gap."),
    ("Information Retrieval low on both platforms",
     "1.7% (Mantle) vs 0.0% (SaaS) — not a Mantle-specific gap. "
     "Mantle additionally limited by a ~40K char context ceiling (500 error above this threshold; "
     "request ID: req_5sukt2z5eweqbcxnm4rxdiuscsxmxq6m6tkt54cxw3ehvxhs7qtq)."),
    ("Exam (Hard) low on both — frontier difficulty",
     "3-4% on both platforms. These are frontier-level problems; "
     "scores are consistent with the model's published performance on comparable hard evals."),
]:
    bullet(doc, title, text)
doc.add_paragraph()

h(doc, "3.7  Pending", level=2)
for item in [
    "HLE on OAI SaaS — pending API quota refresh (GPQA and AIME SaaS completed)",
    "HLE re-score with stronger judge (Claude Sonnet/Opus) for better math equivalence handling",
    "AIME I 2024 — need full 30-problem dataset",
    "Content filter investigation — Q7 block could not be reproduced after initial window",
]:
    bullet(doc, None, item)

doc.save(OUTPUT)
print(f"Saved: {OUTPUT}")
