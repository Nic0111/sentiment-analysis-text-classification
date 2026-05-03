from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer

from txsa_assignment.question_1 import remove_stopwords_and_punctuation, tokenize_with_nltk


@dataclass(frozen=True)
class StemmingResult:
    """Stemmed output from the three required stemming methods."""

    source_tokens: list[str]
    regex_stems: list[str]
    porter_stems: list[str]
    lancaster_stems: list[str]


def load_text(path: Path) -> str:
    """Read a UTF-8 text file and normalize line breaks to spaces."""

    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", " ")


def content_tokens(text: str) -> list[str]:
    """Return punctuation-free content tokens for stemming demonstrations."""

    return remove_stopwords_and_punctuation(tokenize_with_nltk(text)).filtered_tokens


def stem_with_regex(tokens: list[str]) -> list[str]:
    """Stem common suffixes with a regular-expression stemmer."""

    stemmer = RegexpStemmer(r"(ing|ly|ed|ious|ies|ive|es|s|ment)$", min=4)
    return [stemmer.stem(token) for token in tokens]


def stem_with_porter(tokens: list[str]) -> list[str]:
    """Stem tokens with the Porter stemming algorithm."""

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def stem_with_lancaster(tokens: list[str]) -> list[str]:
    """Stem tokens with the Lancaster stemming algorithm."""

    stemmer = LancasterStemmer()
    return [stemmer.stem(token) for token in tokens]


def analyze_text(text: str) -> StemmingResult:
    """Run all stemming methods over the assignment text."""

    tokens = content_tokens(text)
    return StemmingResult(
        source_tokens=tokens,
        regex_stems=stem_with_regex(tokens),
        porter_stems=stem_with_porter(tokens),
        lancaster_stems=stem_with_lancaster(tokens),
    )


def format_section(title: str, value: object) -> str:
    """Format CLI output consistently for report capture."""

    return f"\n## {title}\n{pformat(value, width=100)}"


def build_output(text: str) -> str:
    """Build report-ready stemming output."""

    result = analyze_text(text)
    return "\n".join(
        [
            format_section("Source Tokens", result.source_tokens),
            format_section("Regular Expression Stemmer", result.regex_stems),
            format_section("Porter Stemmer", result.porter_stems),
            format_section("Lancaster Stemmer", result.lancaster_stems),
        ]
    )


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
    """Run the Question 2 analysis and print report-ready output."""

    args = parse_args()
    print(build_output(load_text(args.data_path)))


if __name__ == "__main__":
    main()
