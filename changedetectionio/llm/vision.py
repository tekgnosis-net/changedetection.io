"""Vision support for the LLM call paths.

All multipart message construction, screenshot loading, image preprocessing,
and probe logic lives here. Other modules treat vision as a 'produce messages'
operation and never deal with image bytes directly.

Image preprocessing rationale:
  changedetection.io captures full-page screenshots (default max height ~20000px
  per content_fetchers/__init__.py:SCREENSHOT_MAX_HEIGHT_DEFAULT). Local vision
  encoders have practical input ranges (Qwen3-VL ~ 1280px, Gemma 3 ~ 896px,
  LLaVA 336-672px) and sending oversized images either OOMs the local GPU or
  gets badly downsampled inside the model. We crop + resize before sending.
"""
import base64
import io
import os

from loguru import logger

# Preprocessing defaults — env-overridable for power users.
# Tuned for common local vision models. Override via env vars if your served
# model wants different inputs (e.g. high-detail Qwen3-VL: 1920; LLaVA: 672).
VISION_IMAGE_MAX_WIDTH  = int(os.getenv('VISION_IMAGE_MAX_WIDTH',  1280))
VISION_IMAGE_MAX_HEIGHT = int(os.getenv('VISION_IMAGE_MAX_HEIGHT', 4096))
VISION_IMAGE_MAX_KB     = int(os.getenv('VISION_IMAGE_MAX_KB',      800))
VISION_JPEG_QUALITY     = int(os.getenv('VISION_JPEG_QUALITY',       85))

# Embedded 16x16 PNG used for the capability probe. ~70 bytes after b64.
# This is a deterministic minimal valid PNG (16x16 single-color image).
PROBE_IMAGE_BYTES: bytes = bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000100000001008060000001ff3ff'
    '610000001b49444154789c63fcffff3f0339c0c4c00800000d000100c14a0001'
    '00000049454e44ae426082'
)


class VisionImageTooLargeError(Exception):
    """Image still exceeds size cap after all preprocessing reductions.
    Caller catches this and falls back to text-only path with a warning."""
    pass


def encode_as_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Wrap raw image bytes in OpenAI multipart-format data URL.
    Returns 'data:<mime>;base64,<b64-encoded-bytes>'."""
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def load_screenshot(watch) -> bytes | None:
    """Read <watch.data_dir>/last-screenshot.png. Returns the raw bytes
    or None if the file doesn't exist (e.g. watch never fetched yet, or
    using html_requests fetcher which doesn't render images)."""
    if not getattr(watch, 'data_dir', None):
        return None
    path = os.path.join(watch.data_dir, 'last-screenshot.png')
    if not os.path.isfile(path):
        logger.info(f"vision.load_screenshot: no screenshot at {path}")
        return None
    with open(path, 'rb') as f:
        return f.read()


_HINT_CONTEXT_KEYS = ('model', 'fetcher_backend', 'api_base', 'provider_kind')


def _try_encode(img, quality: int) -> bytes:
    """Encode a PIL Image as JPEG at the given quality."""
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=quality, optimize=True)
    return out.getvalue()


def preprocess_screenshot(image_bytes: bytes,
                          *,
                          hint:       dict | None = None,
                          context:    dict | None = None,
                          max_width:  int = VISION_IMAGE_MAX_WIDTH,
                          max_height: int = VISION_IMAGE_MAX_HEIGHT,
                          max_kb:     int = VISION_IMAGE_MAX_KB,
                          quality:    int = VISION_JPEG_QUALITY) -> tuple[bytes, str, dict]:
    """Resize, top-crop, JPEG-recompress a screenshot for vision-model input.
    Returns (processed_bytes, mime_type='image/jpeg', used_hint).

    Pipeline (top-down on each retry):
      1. Open with PIL; convert to RGB if needed (drops alpha).
      2. Width-cap: width > max_width → downscale proportionally (LANCZOS).
      3. Height-cap: height > max_height → top-crop (above-the-fold kept).
      4. Re-encode as JPEG at `quality`.
      5. If size > max_kb, descend the quality ladder (85 → 75 → 65 → 55).
      6. If still over, scale dimensions ×0.85 and retry (up to 3 dim retries).
      7. If still over, raise VisionImageTooLargeError.

    Hint logic (added in next task) lives at the top of this body.
    """
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Fast-path: try the hinted params first if context matches
    def _hint_context_matches(h, c):
        if not h or not c:
            return False
        return all(h.get(k) == c.get(k) for k in _HINT_CONTEXT_KEYS)

    if _hint_context_matches(hint, context):
        try:
            scaled = img
            hw = hint['max_width']
            hh = hint['max_height']
            hq = hint['quality']
            if scaled.width > hw:
                new_height = int(scaled.height * (hw / scaled.width))
                scaled = scaled.resize((hw, new_height), Image.LANCZOS)
            if scaled.height > hh:
                scaled = scaled.crop((0, 0, scaled.width, hh))
            data = _try_encode(scaled, hq)
            if len(data) <= max_kb * 1024:
                logger.debug(
                    f"vision.preprocess: hint fast-path hit "
                    f"size={scaled.size} q={hq} bytes={len(data)}"
                )
                return data, 'image/jpeg', {
                    'quality': hq, 'max_width': hw, 'max_height': hh,
                    **{k: context[k] for k in _HINT_CONTEXT_KEYS},
                }
        except Exception as e:
            logger.debug(
                f"vision.preprocess: hint fast-path failed ({e}); "
                f"falling back to ladder"
            )

    cur_max_width = max_width
    quality_ladder = [quality, 75, 65, 55]
    cap_bytes = max_kb * 1024
    final_max_height = max_height
    context_fields = {k: (context.get(k) if context else None)
                      for k in _HINT_CONTEXT_KEYS}

    for dim_retry in range(4):
        scaled = img
        if scaled.width > cur_max_width:
            new_height = int(scaled.height * (cur_max_width / scaled.width))
            scaled = scaled.resize((cur_max_width, new_height), Image.LANCZOS)
        if scaled.height > max_height:
            scaled = scaled.crop((0, 0, scaled.width, max_height))

        for q in quality_ladder:
            data = _try_encode(scaled, q)
            if len(data) <= cap_bytes:
                logger.debug(
                    f"vision.preprocess: size={scaled.size} q={q} "
                    f"bytes={len(data)} dim_retry={dim_retry}"
                )
                return data, 'image/jpeg', {
                    'quality': q,
                    'max_width': cur_max_width,
                    'max_height': final_max_height,
                    **context_fields,
                }

        cur_max_width = int(cur_max_width * 0.85)
        if cur_max_width < 256:
            break

    raise VisionImageTooLargeError(
        f"Image still over {max_kb} KB after quality ladder + 3 dim retries"
    )


def load_and_prepare_screenshot(watch, llm_cfg: dict) -> tuple[bytes, str] | None:
    """Load <watch.data_dir>/last-screenshot.png, run preprocess with the
    watch's stored hint and the current run's context.

    Returns (processed_bytes, mime_type) or None on:
      - missing screenshot file
      - PIL decode failure (corrupt or non-image content)
      - VisionImageTooLargeError after all reductions

    On success, persists the new hint (with current context) onto the watch
    via in-place dict mutation. The worker commits the watch at end-of-check
    via the existing watch lifecycle.
    """
    raw = load_screenshot(watch)
    if raw is None:
        return None

    context = {
        'model':           llm_cfg.get('model'),
        'fetcher_backend': watch.get('fetch_backend'),
        'api_base':        llm_cfg.get('api_base'),
        'provider_kind':   llm_cfg.get('provider_kind'),
    }

    try:
        hint = watch.get('llm_vision_preprocess_hint')
        bytes_out, mime, used_hint = preprocess_screenshot(
            raw, hint=hint, context=context
        )
    except VisionImageTooLargeError as e:
        logger.warning(
            f"vision.load_and_prepare: image unrescuable for "
            f"watch={getattr(watch, 'uuid', '?')}: {e}"
        )
        return None
    except Exception as e:
        # PIL decode failure — corrupt or non-image bytes
        logger.warning(
            f"vision.load_and_prepare: decode failure for "
            f"watch={getattr(watch, 'uuid', '?')}: {type(e).__name__}: {e}"
        )
        return None

    if used_hint != hint:
        try:
            watch['llm_vision_preprocess_hint'] = used_hint
        except (TypeError, KeyError):
            pass

    return bytes_out, mime


def build_vision_messages(text_user_content: str,
                          image_bytes: bytes,
                          mime_type: str = 'image/jpeg',
                          system_prompt: str | None = None,
                          previous_screenshot: tuple[bytes, str] | None = None) -> list:
    """Construct OpenAI-format multipart messages list.
    `previous_screenshot` is reserved for a follow-up PR; currently ignored."""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({
        'role': 'user',
        'content': [
            {'type': 'text', 'text': text_user_content},
            {'type': 'image_url', 'image_url': {
                'url': encode_as_data_url(image_bytes, mime_type),
            }},
        ],
    })
    return messages


def probe_vision_capability(model: str,
                            api_key: str | None,
                            api_base: str | None,
                            timeout: int = 30) -> tuple[bool, str]:
    """Send PROBE_IMAGE_BYTES with a trivial 'describe what you see' prompt.
    Return (ok, message_for_user). Used by /settings/llm/vision-test."""
    import litellm
    messages = build_vision_messages(
        text_user_content='Describe what you see in 5 words or fewer.',
        image_bytes=PROBE_IMAGE_BYTES, mime_type='image/png',
    )
    try:
        kwargs = {
            'model': model, 'messages': messages, 'timeout': timeout,
            'temperature': 0, 'max_tokens': 100,
        }
        if api_key:
            kwargs['api_key'] = api_key
        if api_base:
            kwargs['api_base'] = api_base
        response = litellm.completion(**kwargs)
        choice = response.choices[0]
        text = (choice.message.content or '').strip()
        finish = getattr(choice, 'finish_reason', None)
        if not text:
            return False, (
                f"Model responded but returned empty content "
                f"(finish_reason={finish}). The configured model may not "
                f"support image inputs."
            )
        return True, text
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


VISION_FAILURE_STRIKE_LIMIT = 3


def record_vision_failure(watch) -> bool:
    """Increment watch.llm_vision_failure_count. If reaches the strike
    limit, clear watch.llm_vision_verified and reset counter. Returns
    True iff the verified flag was just cleared."""
    count = int(watch.get('llm_vision_failure_count', 0)) + 1
    if count >= VISION_FAILURE_STRIKE_LIMIT:
        watch['llm_vision_verified'] = False
        watch['llm_vision_failure_count'] = 0
        logger.error(
            f"vision: 3 consecutive failures on watch "
            f"{getattr(watch, 'uuid', '?')}; cleared verified flag — "
            f"user must re-probe via 'Test vision capability'"
        )
        return True
    watch['llm_vision_failure_count'] = count
    return False


def reset_vision_failure_count(watch) -> None:
    """Reset the failure counter on any successful vision call."""
    if watch.get('llm_vision_failure_count', 0) > 0:
        watch['llm_vision_failure_count'] = 0
