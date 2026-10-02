"""Shared Claude model IDs and request helpers for the solver.

All Anthropic calls go through ``create_message`` so model choice, refusal
fallback, and response parsing stay consistent across pipeline phases.
"""

from typing import Callable, Optional

from loguru import logger

OPUS_MODEL = "claude-opus-5-5"
SONNET_MODEL = "claude-sonnet-5-5"

# Server-side refusal fallback: on a safety-classifier decline the API re-runs
# the request on the model Anthropic recommends for that refusal category.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {OPUS_MODEL, SONNET_MODEL}


def create_message(client, on_text: Optional[Callable[[str], None]] = None, **kwargs):
    """Call the Messages API and return the final message.

    Requests are streamed: 5.5 models think before answering, so a solve pass
    can run for minutes, and a non-streaming request sits silent on the wire
    that whole time, exposed to dropped connections. Streaming keeps bytes
    flowing; ``get_final_message()`` returns the same object ``create`` would.
    Opus/Sonnet 5.5 requests are also opted into server-side refusal fallback.

    ``on_text`` receives each chunk of answer text as it streams (thinking is
    not included); the live solve view uses it to show answers as they're written.
    """
    if kwargs.get("model") in _FALLBACK_MODELS:
        stream = client.beta.messages.stream(
            betas=[_FALLBACK_BETA], fallbacks="default", **kwargs
        )
    else:
        stream = client.messages.stream(**kwargs)
    with stream as s:
        if on_text is not None:
            for event in s:
                if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                    on_text(event.delta.text)
        return s.get_final_message()


def response_text(response, label: str = "") -> str:
    """Join the response's text blocks, or return "" if the model refused.

    Opus/Sonnet 5.5 responses can open with thinking blocks (and fallback
    blocks), so content must be read by block type, never by position.
    """
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        logger.warning(f"Claude declined {label or 'request'} (category: {category})")
        return ""
    if response.stop_reason == "max_tokens":
        logger.warning(f"{label or 'Request'} hit max_tokens; output is truncated")
    return "\n".join(b.text for b in response.content if b.type == "text").strip()
