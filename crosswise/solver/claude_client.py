"""Shared Claude model IDs and request helpers for the solver.

All Anthropic calls go through ``create_message`` so model choice, refusal
fallback, and response parsing stay consistent across pipeline phases.
"""

from loguru import logger

OPUS_MODEL = "claude-opus-5-5"
SONNET_MODEL = "claude-sonnet-5-5"

# Server-side refusal fallback: on a safety-classifier decline the API re-runs
# the request on the model Anthropic recommends for that refusal category.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = {OPUS_MODEL, SONNET_MODEL}


def create_message(client, **kwargs):
    """Call the Messages API, opting Opus/Sonnet 5.5 requests into refusal fallback."""
    if kwargs.get("model") in _FALLBACK_MODELS:
        return client.beta.messages.create(
            betas=[_FALLBACK_BETA], fallbacks="default", **kwargs
        )
    return client.messages.create(**kwargs)


def response_text(response, label: str = "") -> str:
    """Join the response's text blocks, or return "" if the model refused.

    Opus/Sonnet 5.5 responses can open with thinking blocks (and fallback
    blocks), so content must be read by block type, never by position.
    """
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        logger.warning(f"Claude declined {label or 'request'} (category: {category})")
        return ""
    return "\n".join(b.text for b in response.content if b.type == "text").strip()
