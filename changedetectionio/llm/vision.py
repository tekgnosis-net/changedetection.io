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
import os

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
