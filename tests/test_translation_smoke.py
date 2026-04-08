"""Smoke tests for the gloss_to_text pipeline.

These tests verify that modules import cleanly and that the prompt
strategies dictionary is populated with the expected keys.  No model
weights, no GPU, no external API calls required.
"""

from __future__ import annotations

import pytest  # noqa: F401 — imported for consistency; used if future tests need it


# ---------------------------------------------------------------------------
# 1. Import smoke
# ---------------------------------------------------------------------------

def test_gloss_to_text_imports() -> None:
    """gloss_to_text package-level import must succeed."""
    import gloss_to_text  # noqa: F401


def test_gloss_to_text_prompts_submodule_imports() -> None:
    """gloss_to_text.prompts.strategies must be importable."""
    from gloss_to_text.prompts import strategies  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Prompt strategies
# ---------------------------------------------------------------------------

_EXPECTED_STRATEGY_KEYS: frozenset[str] = frozenset(
    {"P1_EN", "P1_TR", "P2_EN", "P2_TR", "P3_EN", "P3_TR"}
)


def test_prompt_strategies_has_all_keys() -> None:
    """PROMPT_STRATEGIES must contain all six P×_EN/TR keys."""
    from gloss_to_text.prompts.strategies import PROMPT_STRATEGIES

    assert set(PROMPT_STRATEGIES.keys()) == _EXPECTED_STRATEGY_KEYS


def test_prompt_strategies_values_are_non_empty_strings() -> None:
    """Every prompt strategy value must be a non-empty string."""
    from gloss_to_text.prompts.strategies import PROMPT_STRATEGIES

    for key, value in PROMPT_STRATEGIES.items():
        assert isinstance(value, str), f"{key}: expected str, got {type(value)}"
        assert value.strip(), f"{key}: value is blank"


def test_prompt_strategies_dict_is_module_singleton() -> None:
    """Two imports of PROMPT_STRATEGIES must return the same object (module cache)."""
    from gloss_to_text.prompts.strategies import PROMPT_STRATEGIES as a
    from gloss_to_text.prompts.strategies import PROMPT_STRATEGIES as b

    assert a is b


def test_p3_en_strategy_is_present_and_non_trivial() -> None:
    """P3_EN is the best-performing strategy and must have a substantial prompt."""
    from gloss_to_text.prompts.strategies import PROMPT_STRATEGIES

    prompt = PROMPT_STRATEGIES["P3_EN"]
    # Longer than a placeholder — at least 50 characters
    assert len(prompt) >= 50, f"P3_EN prompt suspiciously short: {prompt!r}"
