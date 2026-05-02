"""Shared tokenization helpers for the Part A assignment questions."""

from __future__ import annotations

import re

from nltk.tokenize import TreebankWordTokenizer


def treebank_tokens(text: str) -> list[str]:
    tokenizer = TreebankWordTokenizer()
    sentence_like_chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())

    tokens: list[str] = []
    for chunk in sentence_like_chunks:
        tokens.extend(tokenizer.tokenize(chunk))

    return tokens
