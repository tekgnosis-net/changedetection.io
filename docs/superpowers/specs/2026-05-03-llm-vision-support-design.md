# LLM vision support — design spec

| | |
|---|---|
| **Date** | 2026-05-03 |
| **Status** | Approved (brainstorming → spec); ready for implementation plan |
| **Scope** | Foundational vision support for self-hosted OpenAI-compatible LLM endpoints (vLLM, LM Studio, llama.cpp), wired into two LLM call paths: restock price/stock extraction and `text_json_diff` change summarisation. |
| **Targets fork** | `tekgnosis-net/changedetection.io` (initially) — upstream PR follows once stable, framed as a Phase 2 follow-up to PR #4117 |
| **Builds on** | PR #4117 (OpenAI-compatible provider + token multiplier for reasoning models) — the `provider_kind=='openai_compatible'` flag and `apply_local_token_multiplier` helper introduced there are reused here |
| **Refs** | Issue [#3204](https://github.com/dgtlmoon/changedetection.io/issues/3204) (vLLM integration / vision extraction) — this PR implements the vision aspect explicitly deferred in PR #4117's roadmap section |

---

## Context

The fork now supports self-hosted OpenAI-compatible LLM endpoints (PR #4117). Local vision-capable models (Qwen3-VL, Gemma 3 multimodal, DeepSeek-VL2, etc.) are increasingly common; sending screenshots to them for visual change analysis or product extraction is the natural next step. Issue #3204's discussion explicitly anticipated this work, and PR #4117's roadmap section names it as Phase 2.

**The Playwright finding that reshaped the design**: the watch types most likely to benefit from vision (browser-fetched watches — `restock_diff` in particular, which always uses a browser fetcher) currently see an **empty AI/LLM tab** because of a `{% if is_text_json_diff %}` gate at `templates/edit/include_llm_intent.html:32`. So a meaningful vision feature can't just bolt onto the existing tab; it requires lifting that gate and making the AI tab processor-aware.

---

## Decisions locked (during brainstorming)

| Decision | Locked answer |
|---|---|
| **UI placement** | Lift the `is_text_json_diff` gate; the AI tab becomes processor-aware. `text_json_diff` watches see existing intent + summary + new vision toggle; `restock_diff` watches see vision-only; other processors continue to see no AI content. |
| **MVP scope** | Two call paths: (a) restock vision (price/stock extraction in `processors/restock_diff/plugins/llm_restock.py`), (b) text_json_diff change summary vision (`evaluator.summarise_change`). `evaluate_change` (intent eval) deferred — touches `llm_evaluation_cache` keying. |
| **Provider gate** | `provider_kind == 'openai_compatible'` only. Toggle is greyed-out / disabled for cloud providers (OpenAI, Anthropic, Gemini, OpenRouter) and Ollama. Cloud vision becomes a follow-up if/when there's demand. |
| **Capability check** | Live probe button "Test vision capability" parallels existing "Test connection". Sends a 16×16 embedded probe image with a trivial prompt; on success unlocks the toggle for save; on failure blocks save and shows the error inline. |
| **Screenshot scope** | Current screenshot only (`<watch.data_dir>/last-screenshot.png`). Design includes a deferred `previous_screenshot_path` parameter in `vision.py` signatures so before+after summaries can be added later without changing call sites. |
| **Module structure** | Dedicated `changedetectionio/llm/vision.py` module; new `/settings/llm/vision-test` route in `blueprint/settings/llm.py` paralleling `llm_test`. `client.py` is unchanged (already accepts arbitrary `messages`). |

---

## Architecture

```
                          User opens edit page
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  edit/include_llm_intent.html│
                     │  (gate lifted; processor-    │
                     │   aware sections rendered)   │
                     └──────────────┬───────────────┘
                                    │ AI tab UI
                                    ▼
              ┌─────────────────────────────────────────────┐
              │  Per-watch fields (forms.py + Watch model)  │
              │  • llm_use_vision : bool                    │
              │  • llm_vision_verified : bool (hidden,      │
              │    set by probe; cleared on edit)           │
              └────┬────────────────────────────────────┬───┘
                   │                                    │
            click "Test vision"                  Save & run watch
                   │                                    │
                   ▼                                    ▼
   ┌───────────────────────────────┐    ┌─────────────────────────────┐
   │ /settings/llm/vision-test     │    │ evaluator.summarise_change  │
   │ (new route in blueprint/      │    │ llm_restock.run             │
   │  settings/llm.py)             │    │ check use_vision flag       │
   └──────────────┬────────────────┘    └────────────┬────────────────┘
                  │                                  │
                  ▼                                  ▼
                          ┌─────────────────────────────────┐
                          │      llm/vision.py (NEW)        │
                          │  • load_screenshot              │
                          │  • preprocess_screenshot        │
                          │  • encode_as_data_url           │
                          │  • build_vision_messages        │
                          │  • probe_vision_capability      │
                          │  • PROBE_IMAGE_BYTES (16×16)    │
                          └────────────────┬────────────────┘
                                           │ multipart messages
                                           ▼
                          ┌─────────────────────────────────┐
                          │  llm/client.completion          │
                          │  (UNCHANGED — passes messages   │
                          │   through to litellm)           │
                          └────────────────┬────────────────┘
                                           ▼
                          openai-compatible endpoint (vLLM/LM Studio)
```

**Boundaries:**
- `llm/vision.py` is the only place that knows about images, base64 encoding, multipart message structure, and the probe protocol.
- `llm/client.py` stays unchanged — it already accepts arbitrary `messages`, so vision messages flow through without any signature change.
- The probe endpoint mirrors the existing `llm_test` route exactly — same pattern, same JSON shape, same browser-side button-result UX. Discoverable via existing convention.
- No worker pipeline changes — the existing `last-screenshot.png` lifecycle is sufficient. The deferred `previous_screenshot_path` hook lives as an unused parameter in `build_vision_messages` for a future before+after PR.

---

## Components

### New file: `changedetectionio/llm/vision.py` (~180 LOC)

```python
"""Vision support for the LLM call paths.

All multipart message construction, screenshot loading, image preprocessing,
and probe logic lives here. Other modules treat vision as a 'produce messages'
operation and never deal with image bytes directly.

Image preprocessing rationale:
  changedetection.io captures full-page screenshots (default max height ~20000px
  per content_fetchers/__init__.py:SCREENSHOT_MAX_HEIGHT_DEFAULT). Local vision
  encoders have practical input ranges (Qwen3-VL ≈ 1280px, Gemma 3 ≈ 896px,
  LLaVA 336–672px) and sending oversized images either OOMs the local GPU or
  gets badly downsampled inside the model. We crop + resize before sending.
"""
import os, io, base64
from PIL import Image
from loguru import logger

# ── Preprocessing defaults — env-overridable for power users ──────────────
VISION_IMAGE_MAX_WIDTH  = int(os.getenv('VISION_IMAGE_MAX_WIDTH',  1280))
VISION_IMAGE_MAX_HEIGHT = int(os.getenv('VISION_IMAGE_MAX_HEIGHT', 4096))
VISION_IMAGE_MAX_KB     = int(os.getenv('VISION_IMAGE_MAX_KB',      800))
VISION_JPEG_QUALITY     = int(os.getenv('VISION_JPEG_QUALITY',       85))

# Embedded 16x16 PNG used for the capability probe (~100 bytes).
PROBE_IMAGE_BYTES: bytes = b'\x89PNG\r\n\x1a\n...'  # actual bytes embedded


class VisionImageTooLargeError(Exception):
    """Image still exceeds size cap after all reductions; caller falls back
    to text-only path with a warning."""


def load_screenshot(watch) -> bytes | None:
    """Read <watch.data_dir>/last-screenshot.png. Return None if missing."""

def preprocess_screenshot(image_bytes: bytes,
                          max_width:  int = VISION_IMAGE_MAX_WIDTH,
                          max_height: int = VISION_IMAGE_MAX_HEIGHT,
                          max_kb:     int = VISION_IMAGE_MAX_KB,
                          quality:    int = VISION_JPEG_QUALITY) -> tuple[bytes, str]:
    """Resize, top-crop, JPEG-recompress with quality + dim ladder.
    Returns (processed_bytes, mime_type='image/jpeg').
    Raises VisionImageTooLargeError if unrescuable after 3 retries."""

def encode_as_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Wrap preprocessed bytes in OpenAI-format data URL."""

def load_and_prepare_screenshot(watch) -> tuple[bytes, str] | None:
    """Convenience: load_screenshot + preprocess_screenshot.
    Returns None on missing file, decode failure, or VisionImageTooLargeError
    (logs WARN). Caller falls back to text-only path on None."""

def build_vision_messages(text_user_content: str,
                          image_bytes: bytes,
                          mime_type: str = 'image/jpeg',
                          system_prompt: str | None = None,
                          previous_screenshot: tuple[bytes, str] | None = None
                          ) -> list[dict]:
    """OpenAI-format multipart messages list. previous_screenshot reserved
    for follow-up; currently unused if not None."""

def probe_vision_capability(model: str,
                            api_key: str | None,
                            api_base: str | None,
                            timeout: int = 30) -> tuple[bool, str]:
    """Send PROBE_IMAGE_BYTES with 'describe what you see' prompt.
    Returns (ok, message_for_user). Used by /settings/llm/vision-test."""
```

### Touched files

| File | Change | LOC |
|---|---|---|
| `forms.py` | Add `BooleanField llm_use_vision` + `HiddenField llm_vision_verified` to per-watch form. | +15 |
| `blueprint/settings/llm.py` | New `/vision-test` route paralleling `llm_test`. Calls `vision.probe_vision_capability` and returns `{ok, error, text, tokens}` JSON. | +25 |
| `templates/edit/include_llm_intent.html` | Drop the `{% if is_text_json_diff %}` gate at line 32. Branch into processor-aware sections: text_json_diff (intent + summary + vision), restock_diff (vision-only), other (no content). Add the vision section template with toggle, "Test vision capability" button, result panel, cloud-provider greying logic. | +30 / -1 |
| `llm/evaluator.py:summarise_change` | Read `watch.llm_use_vision` + `watch.llm_vision_verified`. If both set + `provider_kind == 'openai_compatible'` + fetcher is browser-based, call `vision.load_and_prepare_screenshot` + `vision.build_vision_messages` and pass to `client.completion`. Fall through to existing text path on `None`. | +15 |
| `processors/restock_diff/plugins/llm_restock.py` | Same gate + load + build pattern. Different system prompt (existing restock prompt + "look at this image" addition). Returns same `{price, currency, availability}` JSON shape — same parser, same caller. | +15 |
| `model/__init__.py` | Add `'llm_use_vision': False`, `'llm_vision_verified': False` to watch base dict. | +2 |

### New test files

| File | Purpose | LOC |
|---|---|---|
| `tests/llm/test_vision.py` | Unit tests for `vision.py` in isolation; mocks `litellm.completion` for probe tests. | ~80 |
| `tests/llm/test_vision_restock.py` | Integration test: restock vision call path produces multipart messages + parses response. | ~30 |
| `tests/llm/test_vision_summary.py` | Integration test: text_json_diff change-summary vision call path. | ~30 |
| `tests/llm/test_vision_endpoint.py` | Integration test for `/settings/llm/vision-test` route. | ~25 |
| `tests/llm/test_vision_regression.py` | **Load-bearing**: vision-off watches produce byte-identical `messages` to today's behavior. | ~20 |

### Translation deltas

3–5 new English msgids (BooleanField label, button label, probe-result strings, vision section help). Standard `python setup.py extract_messages → update_catalog → compile_catalog` triple. Propagates to 14 `.po` catalogs.

### Total estimated footprint

**~340–420 LOC of code + ~185 LOC of tests = ~525–605 LOC end-to-end** across 6 changed files + 1 new module + 5 new test files. Single-PR scope; comparable to PR #4117 in size.

---

## Data flow

### (A) Edit page — toggle, probe, save

1. User opens `/edit/<uuid>#ai-llm`. Template renders processor-aware AI section. `llm_use_vision` and `llm_vision_verified` are populated from the watch record.
2. User ticks the vision toggle. JS clears `llm_vision_verified` (any change to model/api_base in the global LLM settings would also clear it).
3. User clicks **Test vision capability**. Browser fetches `GET /settings/llm/vision-test`. Server calls `vision.probe_vision_capability(model, api_key, api_base)` which builds a multipart request with `PROBE_IMAGE_BYTES`, calls `litellm.completion`, captures response.
4. JSON `{ok, text, tokens}` returned. JS displays inline result. On `ok=true`, JS sets `llm_vision_verified=1` and unlocks save.
5. User saves form. `form.validate()` rejects save if `llm_use_vision=True` AND `llm_vision_verified=False`. Otherwise persists both flags into the watch record.

### (B) Watch run — restock extraction with vision

1. Worker runs the restock_diff processor on a watch with vision enabled.
2. Existing structured-data extraction (JSON-LD / microdata) runs first — if successful, vision path is skipped (this PR doesn't change that priority).
3. If structured data is missing, fall through to the LLM fallback path in `llm_restock.py`. Check `watch.llm_use_vision AND watch.llm_vision_verified AND llm_cfg.provider_kind == 'openai_compatible'`.
4. If gated on, call `vision.load_and_prepare_screenshot(watch)` — opens `last-screenshot.png`, runs preprocessing (resize, top-crop, JPEG q=85, quality ladder if oversize). Returns `(bytes, mime)` or `None`.
5. If `None` (missing screenshot, decode failure, or unrescuable size), fall through to existing text-only path. INFO/WARN logged.
6. Otherwise, `vision.build_vision_messages(user_prompt, image_bytes, mime, system=SYSTEM_PROMPT)` returns multipart list.
7. `client.completion(model, messages, api_key, api_base, max_tokens=apply_local_token_multiplier(80, llm_cfg))` issues the call. Same `apply_local_token_multiplier` introduced in PR #4117.
8. Response JSON parsed normally; same `{price, currency, availability}` shape; same downstream consumers.

### (C) Watch run — text_json_diff change summary with vision

1. Worker detects a change on a text_json_diff watch with vision enabled.
2. Existing text-diff is computed; user's `llm_change_summary` prompt resolved via cascade.
3. Check `watch.llm_use_vision AND watch.llm_vision_verified AND llm_cfg.provider_kind == 'openai_compatible' AND fetcher_is_browser_based(watch)`. The fetcher gate is new for this path: `fetcher_is_browser_based` returns `True` iff `watch.get('fetch_backend')` is one of `html_webdriver`, `html_playwright`, `html_puppeteer` (or any pluggy-registered fetcher whose class is a `Fetcher` subclass that produces actual screenshots — `html_requests` is excluded because its `self.screenshot` field holds the raw HTTP response body, not an image).
4. `vision.load_and_prepare_screenshot(watch)` → `(bytes, mime)` or `None`.
5. If non-`None`, `build_vision_messages(diff_text, image_bytes, mime, system=summary_prompt + image_hint)` returns multipart list.
6. If `None`, fall through to existing text-only `messages` construction.
7. `client.completion(messages, max_tokens=apply_local_token_multiplier(_summary_max_tokens(diff), llm_cfg))`. Returns summary text.
8. `accumulate_global_tokens(...)` updates token counters. Summary returned to caller; substituted into `{{ diff }}` token in notification template.

### Key state transitions

| Field | When set | When cleared | Effect |
|---|---|---|---|
| `watch.llm_use_vision` | Form save with toggle on | Form save with toggle off | Gates the vision call paths |
| `watch.llm_vision_verified` | JS sets after probe success | JS clears on global model/api_base change; form rejects save with vision-on but unverified | Prevents save with broken setup |
| `last-screenshot.png` (per-watch) | Worker on every successful browser fetch (existing) | Watch deletion (existing) | Source data |
| `llm_evaluation_cache` (existing on `evaluate_change`) | Unchanged in this PR | Unchanged | MVP doesn't touch the cached call path; relevant only if vision is later wired into `evaluate_change` |

---

## Error handling

Vision is a **best-effort enhancement that never breaks an existing watch run**. Every vision-prep failure mode falls back to the same text-only behavior the watch would have had with vision off.

| # | Failure | Handling | Log level |
|---|---|---|---|
| 1 | Screenshot file missing (fresh watch, html_requests fetcher) | `load_and_prepare_screenshot` returns `None`; fall through to text-only path | INFO |
| 2 | Screenshot file corrupt or non-image (PIL decode failure) | Same — `None` returned, fall through | WARNING |
| 3 | Image still oversized after preprocessing (`VisionImageTooLargeError`) | Caught in `load_and_prepare_screenshot`, returns `None` with final-attempt dimensions logged | WARNING |
| 4 | Endpoint rejects multipart at runtime (model swapped, capability drift) | Caught at call site, fall through to text-only. **3 consecutive failures clear `watch.llm_vision_verified`** so the toggle is invalidated and the user is prompted to re-probe. | ERROR (on 3rd) |
| 5 | Probe endpoint network error during Test vision click | `probe_vision_capability` returns `(False, str(error))`; UI shows error inline; toggle stays unverified, save-locked. | (browser side) |

### Behavior contract

Vision is a *parallel* path next to existing logic, never a *replacement*. The text-only fallback exists for every vision call site (it's literally what the code did pre-this-PR). All errors caught at the boundary between vision-prep and the LLM call. **Existing tests stay green for vision-off watches** — guaranteed by `test_vision_regression.py`.

---

## Testing

### Test files (unit + integration)

- **`tests/llm/test_vision.py`** — 14 unit tests covering: preprocessing (normal, full-page top-crop, quality ladder, dimension shrink, unrescuable, RGBA), `build_vision_messages` shape, `probe_vision_capability` success/rejection/timeout, `load_screenshot` missing/corrupt cases.
- **`tests/llm/test_vision_restock.py`** — integration: configures a restock watch with vision flags + planted screenshot; mocks `litellm.completion`; asserts captured `messages` is multipart and the JSON parse downstream still works.
- **`tests/llm/test_vision_summary.py`** — integration: text_json_diff watch + browser fetcher + vision flags; asserts captured `messages` is multipart with system prompt mentioning visual context; asserts `{{ diff }}` resolves to mocked summary.
- **`tests/llm/test_vision_endpoint.py`** — integration: `/settings/llm/vision-test` route with mocked `litellm.completion` for success and failure paths.
- **`tests/llm/test_vision_regression.py`** — load-bearing invariant: vision-off watches produce byte-identical `messages` (no `image_url` entries) compared to today.

### What's not tested (with rationale)

| Concern | Why skipped |
|---|---|
| Real local-LLM call against a live vLLM endpoint | No CI infra; flaky |
| UI rendering (toggle, probe button, result panel) | Project has minimal frontend tests; manual verification covers it |
| PIL decode edge cases | Pillow's own test suite |
| Vision-token cost / billing | Out of scope for MVP |
| Concurrent vision-call race on `last-screenshot.png` mid-write | Existing atomic-write semantics; deferred file-lock test |

### Manual verification checklist (post-implementation)

1. Open the **Anaconda restock watch** → AI/LLM tab → confirm vision section now appears (gate lifted).
2. Tick "Use vision for price/stock extraction" → confirm verified flag clears.
3. Click "Test vision capability" → confirm probe succeeds, toggle save-unlocks.
4. Save → run watch → check logs for `vision.preprocess_screenshot` debug lines + multipart `litellm.completion` call.
5. Confirm restock state populates correctly.
6. Switch served vLLM model to a non-vision one (e.g. Qwen3-32B-Instruct) → run watch → after 3 strikes, confirm vision_verified clears + UI shows "re-probe needed".
7. Open a **text_json_diff watch with browser fetcher** (would need to be created) → confirm vision toggle works for change summaries.
8. Open a **text_json_diff watch with html_requests fetcher** (Hacker News) → confirm vision toggle is greyed out with tooltip "needs a browser fetcher".

---

## Out of scope (deferred)

| Item | Why deferred |
|---|---|
| Vision for `evaluate_change` (intent eval) | Would require updating `llm_evaluation_cache` key to include screenshot identity (e.g. screenshot file sha256). Real value, but a deeper PR. Easy follow-up once foundation lands. |
| Cloud-provider vision (GPT-4o, Claude 3+, Gemini 1.5+) | Need cost-warning UX + provider-specific capability detection. Follow-up if/when there's demand. |
| Before+after vision summaries | Requires worker pipeline change (preserve previous screenshot before overwrite). `previous_screenshot_path` parameter already reserved in `build_vision_messages` for this. |
| Tiled vision for full-page screenshots | Top-crop is the MVP simplification. Tiling is well-documented for vision models but adds 5× token cost and more prompt engineering. Revisit if real users hit the below-the-fold limitation. |
| Per-watch vision quality override (low/auto/high) | Env vars are sufficient power-user knob for MVP. UI override is straightforward to add later if requested. |
| Surfacing `finish_reason='length'` from `client.completion` (PR #4117 audit P0) | Independent concern; would change return tuple shape across codebase. Same architectural shape it had before this PR. |

---

## Implementation order (rough)

For the eventual `writing-plans` skill phase. Each step is independently mergeable as a single commit; the implementation plan will refine into explicit milestones.

1. **`vision.py` skeleton + unit tests** — module with all functions stubbed; tests pass for preprocess, build_messages, probe (mocked litellm).
2. **`/settings/llm/vision-test` route** + endpoint integration test.
3. **Form + Watch model fields** — `llm_use_vision`, `llm_vision_verified`. Defaults False. Persisted through form save.
4. **Template lift the gate + add vision section** — `include_llm_intent.html` becomes processor-aware.
5. **Browser-side JS** — toggle interaction, probe button, result rendering, verified-flag clearing on edit.
6. **Wire `summarise_change`** — vision branch with fall-through; integration test.
7. **Wire `llm_restock`** — vision branch with fall-through; integration test.
8. **Regression test** — assert vision-off path is byte-identical.
9. **3-strikes invalidation** — counter on watch, error-handler clears `llm_vision_verified`. Hard to integration-test cleanly; manual verification sufficient.
10. **Translations** — extract / update / compile.
11. **README + docs/** — Vision configuration subsection (env vars, model gating, manual verification).

---

## Notes for the upstream PR (eventual)

When this lands stable on the fork and we prepare the upstream PR:
- Title: `LLM - Vision support for self-hosted OpenAI-compatible endpoints (Phase 2 of #4117) — refs #3204`
- Body: similar structure to PR #4117, with this spec as the design reference.
- Scope discipline (per the saved memory): only LLM code + templates + translations + README. No workflow / CI changes.
- Reference PR #4117 explicitly so the maintainer can read the foundation alongside.
