"""Question 1: tokenization and filtering for ``Data_1.txt``.

The module is intentionally small and explicit so the assignment logic is easy
to inspect, run, and explain in the report.
"""

from __future__ import annotations

import argparse
import re
import string
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import nltk
from nltk.data import find
from nltk.tokenize import word_tokenize


ASSIGNMENT_STOPWORDS = frozenset(
    {
        "a",
        "all",
        "an",
        "and",
        "are",
        "be",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "may",
        "not",
        "of",
        "other",
        "the",
        "to",
    }
)
PUNKT_RESOURCE = "tokenizers/punkt_tab"
PUNKT_DOWNLOAD = "punkt_tab"


@dataclass(frozen=True)
class FilteringResult:
    """Tokens kept after filtering and stop words found in the source text."""

    filtered_tokens: list[str]
    stopwords_found: list[str]


def load_text(path: Path) -> str:
    """Read a UTF-8 text file and normalize line breaks to spaces."""

    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", " ")


def tokenize_with_split(text: str) -> list[str]:
    """Tokenize by splitting only on whitespace."""

    return text.split()


def tokenize_with_regex(text: str) -> list[str]:
    """Tokenize into words and punctuation with a regular expression."""

    return re.findall(r"\b\w+\b|[^\w\s]", text)


def ensure_punkt_available() -> None:
    """Download NLTK Punkt data if word tokenization cannot find it."""

    try:
        find(PUNKT_RESOURCE)
    except LookupError:
        nltk.download(PUNKT_DOWNLOAD, quiet=True)


def tokenize_with_nltk(text: str) -> list[str]:
    """Tokenize with NLTK's word tokenizer using Punkt sentence rules."""

    ensure_punkt_available()
    return word_tokenize(text)


def get_stopwords() -> set[str]:
    """Return the fixed stop-word list used for this assignment."""

    return set(ASSIGNMENT_STOPWORDS)


def remove_stopwords_and_punctuation(tokens: list[str]) -> FilteringResult:
    """Remove English stop words and punctuation from token boundaries."""

    stopwords = get_stopwords()
    punctuation = set(string.punctuation)

    filtered_tokens: list[str] = []
    stopwords_found: set[str] = set()

    for token in tokens:
        cleaned_token = token.strip(string.punctuation)
        normalized = cleaned_token.casefold()

        if normalized in stopwords:
            stopwords_found.add(normalized)
            continue

        if not cleaned_token or all(character in punctuation for character in cleaned_token):
            continue

        filtered_tokens.append(cleaned_token)

    return FilteringResult(
        filtered_tokens=filtered_tokens,
        stopwords_found=sorted(stopwords_found),
    )


def format_section(title: str, value: object) -> str:
    """Format CLI output consistently for report capture."""

    return f"\n## {title}\n{pformat(value, width=100)}"


def build_output(text: str) -> str:
    """Build the full Question 1 output from the provided text."""

    split_tokens = tokenize_with_split(text)
    regex_tokens = tokenize_with_regex(text)
    nltk_tokens = tokenize_with_nltk(text)
    filtering_result = remove_stopwords_and_punctuation(nltk_tokens)

    sections = [
        "## Raw Text\n" + text,
        format_section("Split Tokenization", split_tokens),
        format_section("Regular Expression Tokenization", regex_tokens),
        format_section("NLTK Tokenization", nltk_tokens),
        format_section("Stop Words Found", filtering_result.stopwords_found),
        format_section("Filtered Tokens", filtering_result.filtered_tokens),
    ]

    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=Path("data/Data_1.txt"),
        help="Path to Data_1.txt.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Question 1 analysis and print report-ready output."""

    args = parse_args()
    text = load_text(args.data_path)
    print(build_output(text))


if __name__ == "__main__":
    main()
