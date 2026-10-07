# Specification: feedback-sentiment-classifier-agent

> **Guidelines**: Read all applicable guidelines before executing ANY tasks below:
> - [guidelines.md](../guidelines.md) — Universal execution rules
> - [guidelines-agent.md](../guidelines-agent.md) — Universal agent patterns
> - [guidelines-agent-python.md](../guidelines-agent-python.md) — Python implementation details
> - [guidelines-agent-skills.md](../guidelines-agent-skills.md) — Runtime skills patterns
> - [guidelines-agent-mcp.md](../guidelines-agent-mcp.md) — MCP integration patterns

---

## Basic Setup

- [x] Read `product-requirements-document.md` and `intent.md` at the solution root.
- [x] Bootstrap agent code in `assets/feedback-sentiment-classifier-agent/` using instructions from the `sap-agent-bootstrap` section. (invoke from inside `assets/feedback-sentiment-classifier-agent/`, use copy commands — do NOT create files manually)
- [x] Install dependencies; validate the agent starts and responds at `/.well-known/agent.json`.

---

## Runtime Skills

No runtime skills are required. The classification logic is simple enough to be handled entirely in the system prompt and agent code without branching workflows or reference material.

---

## Project-Specific Tasks

### Sentiment Classification Core

- [x] In `app/agent.py`, implement the system prompt instructing the LLM to:
  - Classify the input feedback text as exactly one of: `positive`, `negative`, or `neutral`.
  - Return a confidence score as a float between 0.0 and 1.0.
  - Return a reasoning explanation in at most 3 sentences.
  - Never fabricate or infer information beyond the given feedback text.
  - If confidence < 0.5, include `"low_confidence": true` in the response.
- [x] Implement a structured output schema for the agent response containing:
  - `sentiment` (string: `"positive"` | `"negative"` | `"neutral"`)
  - `confidence` (float: 0.0–1.0)
  - `reasoning` (string: non-empty, ≤ 3 sentences)
  - `low_confidence` (boolean: `true` if confidence < 0.5, otherwise `false`)
- [x] Ensure the agent accepts a single `feedback_text` string as input via the A2A protocol request.
- [x] Wire the LLM call through LiteLLM (SAP Generative AI Hub) — use `sap/openai--gpt-4o-mini` as the default model to satisfy the < 1 second SLA.
- [x] Validate that the LLM response is parsed into the structured output schema before returning; handle malformed LLM responses gracefully with a structured error response.

### Agent Extensibility

- [x] Expose a pre-processing extension point in `app/agent.py`: a hook or overridable method `preprocess_feedback(text: str) -> str` that callers can override to apply custom transformations (e.g., language detection, PII scrubbing) before the LLM call.
- [x] Expose a post-processing extension point: an overridable method `postprocess_result(result: dict) -> dict` that callers can override to remap labels or enrich the result after the LLM call.

### Error Handling & Guardrails

- [x] If the LLM inference fails or times out, return a structured JSON error response (do NOT raise an unhandled exception):
  ```json
  { "error": "<error message>", "sentiment": null, "confidence": null, "reasoning": null }
  ```
- [x] Never log or store the raw `feedback_text` input (PII risk). Log only metadata (e.g., input length, processing duration).
- [x] Add a response-time check: log a warning if the agent's total processing time exceeds 900 ms.

---

## Business Instrumentation

- [x] Implement structured logging and OpenTelemetry spans for each milestone:

  | Milestone | Achievement log | Miss log |
  |-----------|----------------|----------|
  | M1 Input Received | `M1.achieved: feedback text input received and validated` | `M1.missed: feedback text input missing or invalid` |
  | M2 Sentiment Classified | `M2.achieved: sentiment classified as {sentiment}` | `M2.missed: sentiment classification did not return a valid label` |
  | M3 Reasoning Provided | `M3.achieved: reasoning explanation generated` | `M3.missed: reasoning explanation could not be generated` |
  | M4 Response Delivered | `M4.achieved: structured response delivered within SLA` | `M4.missed: response delivery exceeded SLA or failed` |

- [x] Extract all business logic from `stream()` into a plain async helper method `_run_classification()` and apply `@tracer.start_as_current_span` on that method — never inside the async generator (`stream()`).
- [x] Verify `bootstrap(app)` is called after `app = server.build()` in `main.py`.

---

## MCP Tool Integration

This agent has no SAP API dependencies. All intelligence comes from the LLM via SAP Generative AI Hub. No MCP server creation, no API discovery, no `mcp-translation-file` invocation, and no `mcp-mock.json` generation are required.

---

## Testing

- [x] Ensure `conftest.py` only sets `IBD_TESTING=true`.
- [x] Write unit tests in `assets/feedback-sentiment-classifier-agent/tests/`:
  - `test_classify_positive.py` — verify positive feedback returns `sentiment: "positive"` with confidence > 0 and non-empty reasoning (mock LLM).
  - `test_classify_negative.py` — verify negative feedback returns `sentiment: "negative"` (mock LLM).
  - `test_classify_neutral.py` — verify neutral feedback returns `sentiment: "neutral"` (mock LLM).
  - `test_low_confidence_flag.py` — verify that a mock LLM response with confidence < 0.5 sets `low_confidence: true`.
  - `test_llm_failure.py` — verify that an LLM timeout/error returns a structured error response and does NOT raise an exception.
  - `test_preprocessing_hook.py` — verify that `preprocess_feedback` is called before the LLM and that overriding it changes the input.
  - `test_postprocessing_hook.py` — verify that `postprocess_result` is called after the LLM and that overriding it changes the output.
- [x] Write one integration test `test_end_to_end.py` exercising the full agent flow: send a feedback text via A2A invoke, mock the LLM response, assert structured output fields (`sentiment`, `confidence`, `reasoning`, `low_confidence`).
- [x] Run `pytest` from `assets/feedback-sentiment-classifier-agent/` (no args). If coverage < 70%, add targeted tests.
- [x] Run `pytest` again from `assets/feedback-sentiment-classifier-agent/` (no args) to generate final `test_report.json`.
- [x] Verify `test_report.json` exists in `assets/feedback-sentiment-classifier-agent/`.
