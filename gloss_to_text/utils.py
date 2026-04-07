"""
Shared utilities for the gloss-to-text pipeline.

Centralises Turkish-aware text helpers and model-specific chat template
formatting that were previously duplicated across every benchmark script.
"""

import re


# ---------------------------------------------------------------------------
# Turkish text helpers
# ---------------------------------------------------------------------------

def turkish_lower(text: str) -> str:
    """Lowercase with correct Turkish İ→i and I→ı mappings."""
    if not text:
        return ""
    return text.replace("İ", "i").replace("I", "ı").lower()


def turkish_capitalize(text: str) -> str:
    """Capitalise the first character with Turkish i/ı rules."""
    if not text:
        return ""
    first = text[0]
    if first == "i":
        return "İ" + text[1:]
    if first == "ı":
        return "I" + text[1:]
    return first.upper() + text[1:]


def polish_turkish(text: str) -> str:
    """
    Post-process a model prediction for Turkish output:
    - Capitalise the first letter (Turkish-aware).
    - Capitalise after sentence-ending punctuation.
    - Capitalise proper nouns followed by an apostrophe.
    - Ensure the text ends with sentence-ending punctuation.
    """
    text = text.strip()
    if not text:
        return ""

    text = turkish_capitalize(text)

    def _cap_match(match: re.Match) -> str:
        return match.group(1) + turkish_capitalize(match.group(2))

    text = re.sub(r"([.!?]\s+)([a-zığüşöç])", _cap_match, text)

    def _cap_proper(match: re.Match) -> str:
        return turkish_capitalize(match.group(0))

    text = re.sub(r"\b[a-zığüşöç]+'[a-zığüşöç]*\b", _cap_proper, text)

    if text[-1] not in ".!?":
        text += "."

    return text


# ---------------------------------------------------------------------------
# Chat template formatting
# ---------------------------------------------------------------------------

def get_chat_template(
    model_name: str,
    instruction: str,
    gloss: str,
    output: str = "",
) -> str:
    """
    Return a model-specific chat-formatted prompt string.

    Supports Gemma, Llama-3, and Trendyol (ChatML) formats.
    The optional *output* argument is used during training to append the
    expected response; leave it empty for inference.
    """
    m = model_name.lower()
    if "gemma" in m:
        return (
            f"<start_of_turn>user\n{instruction}\n\nGloss: {gloss}"
            f"<end_of_turn>\n<start_of_turn>model\n{output}"
        )
    if "llama" in m:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}\n\nGloss: {gloss}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}"
        )
    # Trendyol (ChatML)
    return (
        f"<|im_start|>user\n{instruction}\n\nGloss: {gloss}"
        f"<|im_end|>\n<|im_start|>assistant\n{output}"
    )
