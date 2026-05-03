# AI Vision Configuration

When using the **OpenAI-compatible (vLLM, LM Studio, llama.cpp)** LLM provider with a vision-capable local model (Qwen3-VL, Gemma 3 multimodal, LLaVA variants, etc.), per-watch vision controls appear in the AI tab.

## Per-watch toggles

For **text_json_diff** watches using a browser-based fetcher (Playwright / Puppeteer / WebDriver):
- **Send the page screenshot to the LLM** — when a change is detected, sends the latest screenshot alongside the text diff to the AI Change Summary call.

For **restock_diff** watches:
- **Use the LLM for price/stock extraction** — overrides the global `llm_restock_use_fallback_extract` per-watch (Inherit / On / Off).
- **Send the page screenshot to the LLM** (vision sub-toggle) — when on, the LLM is fed the screenshot in addition to the page text. Falls back gracefully to text-only LLM extraction if vision can't be used.

## Capability probe

Click **Test vision capability** to send a tiny embedded probe image to the configured local endpoint. On success, the toggle unlocks for save. On failure, the toggle stays locked and the inline error explains why.

## Image preprocessing env vars

| Variable | Default | Description |
|---|---|---|
| `VISION_IMAGE_MAX_WIDTH` | `1280` | Width cap (px). |
| `VISION_IMAGE_MAX_HEIGHT` | `4096` | Height cap (px). For full-page screenshots taller than this, the image is **top-cropped**. |
| `VISION_IMAGE_MAX_KB` | `800` | File size cap (KB). |
| `VISION_JPEG_QUALITY` | `85` | Initial JPEG quality. |

## Caching

After the first successful vision call on a watch, the winning preprocess parameters and run context (`{model, fetcher_backend, api_base, provider_kind}`) are persisted as `llm_vision_preprocess_hint`. Subsequent runs fast-path through the hinted params. Hint invalidates implicitly when context drifts.

## Failure handling

Vision is **best-effort, never breaks watch runs**:
- Missing screenshot, corrupt image, oversize-after-preprocessing → silent fall-through to text-only.
- Endpoint rejects multipart at runtime → fall through, log warning. **3 consecutive failures** clear `llm_vision_verified` — the user must re-probe.

## Models known to work

- **Qwen3-VL** family (1B / 4B / 8B / 30B-A3B / 32B / 235B-A22B-Instruct on vLLM)
- **Gemma 3** multimodal (4B / 12B / 27B)
- **LLaVA** variants (1.5 / 1.6 / OneVision)
- **DeepSeek-VL2** (Tiny / Small / Base)

Models that are **not vision-capable** (probe will fail):
- Qwen3-32B-Instruct, DeepSeek-R1-Distill-*, most plain `*-Instruct` variants.
