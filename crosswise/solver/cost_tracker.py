"""
Thread-safe API cost tracker for crossword solver.

Tracks input/output tokens and costs across all API calls during a solve,
then prints a summary breakdown by phase.

Accounts for:
- Base input/output token costs
- Cache write tokens (1.25x base input) and cache read tokens (0.1x base input)
- Web search per-query costs ($0.01/search)
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


# Pricing per million tokens: (input_cost, output_cost)
MODEL_PRICING: Dict[str, tuple] = {
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

# Web search: $10 per 1,000 searches
WEB_SEARCH_COST_PER_QUERY = 0.01


@dataclass
class APICall:
    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    web_searches: int
    cost: float


class CostTracker:
    """Accumulates API costs across all calls during a solve."""

    def __init__(self):
        self._calls: List[APICall] = []
        self._lock = threading.Lock()

    def track(self, response, label: str, model: Optional[str] = None) -> float:
        """Track an Anthropic API response's token usage and cost.

        Args:
            response: Anthropic API response object with .usage and .model
            label: Human-readable label for this call (e.g., "solve_pass_1")
            model: Override model name (uses response.model if not provided)

        Returns:
            Cost of this call in dollars.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0

        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        model_id = model or getattr(response, "model", "unknown")

        # Web search count from server_tool_use
        web_searches = 0
        server_tool_use = getattr(usage, "server_tool_use", None)
        if server_tool_use:
            web_searches = getattr(server_tool_use, "web_search_requests", 0) or 0

        # Look up pricing — try exact match, then prefix match
        pricing = MODEL_PRICING.get(model_id)
        if pricing is None:
            for key, val in MODEL_PRICING.items():
                if key.startswith(model_id.split("-20")[0]) or model_id.startswith(key.split("-20")[0]):
                    pricing = val
                    break
        if pricing is None:
            pricing = (15.0, 75.0)  # default to Opus pricing (conservative)

        input_cost_per_m, output_cost_per_m = pricing
        cost = (
            input_tokens / 1_000_000 * input_cost_per_m +
            output_tokens / 1_000_000 * output_cost_per_m +
            cache_write_tokens / 1_000_000 * input_cost_per_m * 1.25 +
            cache_read_tokens / 1_000_000 * input_cost_per_m * 0.1 +
            web_searches * WEB_SEARCH_COST_PER_QUERY
        )

        call = APICall(
            label=label,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            web_searches=web_searches,
            cost=cost,
        )

        with self._lock:
            self._calls.append(call)

        # Short model name for display
        short_model = "opus" if "opus" in model_id else "sonnet" if "sonnet" in model_id else "haiku" if "haiku" in model_id else model_id

        # Build extras string
        extras = []
        if cache_write_tokens:
            extras.append(f"{cache_write_tokens:,} cache_write")
        if cache_read_tokens:
            extras.append(f"{cache_read_tokens:,} cache_read")
        if web_searches:
            extras.append(f"{web_searches} search{'es' if web_searches > 1 else ''}")
        extra_str = f" + {', '.join(extras)}" if extras else ""

        logger.debug(f"[cost] ${cost:.4f}  {label}: {input_tokens:,} in / {output_tokens:,} out{extra_str} ({short_model})")

        return cost

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(c.cost for c in self._calls)

    def summary(self) -> str:
        """Return a formatted cost summary grouped by label prefix."""
        with self._lock:
            calls = list(self._calls)

        if not calls:
            return "=== No API calls tracked ==="

        # Group by phase (label prefix before underscore+digit or full label)
        phases: Dict[str, List[APICall]] = {}
        for call in calls:
            # Extract phase: "solve_pass_1" → "solve_pass", "web_prepass_3" → "web_prepass"
            parts = call.label.rsplit("_", 1)
            if parts[-1].isdigit() and len(parts) > 1:
                phase = parts[0]
            else:
                phase = call.label
            phases.setdefault(phase, []).append(call)

        total = sum(c.cost for c in calls)
        total_searches = sum(c.web_searches for c in calls)
        lines = [f"=== Total API Cost: ${total:.2f} ==="]
        for phase, phase_calls in phases.items():
            phase_cost = sum(c.cost for c in phase_calls)
            searches = sum(c.web_searches for c in phase_calls)
            search_str = f", {searches} searches" if searches else ""
            lines.append(f"  {phase:30s} ${phase_cost:.3f}  ({len(phase_calls)} calls{search_str})")

        if total_searches:
            lines.append(f"  {'(includes web search fees)':30s} ${total_searches * WEB_SEARCH_COST_PER_QUERY:.2f}  ({total_searches} searches @ ${WEB_SEARCH_COST_PER_QUERY})")

        return "\n".join(lines)


# Global tracker instance — reset per solve
_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Get the global cost tracker, creating one if needed."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def reset_tracker() -> CostTracker:
    """Reset and return a fresh global cost tracker."""
    global _tracker
    _tracker = CostTracker()
    return _tracker
