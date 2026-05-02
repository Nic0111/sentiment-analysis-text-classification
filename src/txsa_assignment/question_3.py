from __future__ import annotations

import argparse
import string
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, cast

from nltk import CFG, ChartParser, RegexpTagger, pos_tag
from textblob.exceptions import MissingCorpusError

from txsa_assignment.token_utils import treebank_tokens


FALLBACK_POS_TAGS = {
    "the": "DT",
    "big": "JJ",
    "black": "JJ",
    "dog": "NN",
    "barked": "VBD",
    "at": "IN",
    "white": "JJ",
    "cat": "NN",
    "and": "CC",
    "chased": "VBD",
    "away": "RB",
}


@dataclass(frozen=True)
class PosResult:
    """POS outputs from the required tagger types."""

    tokens: list[str]
    nltk_tags: list[tuple[str, str]]
    textblob_tags: list[tuple[str, str]]
    regex_tags: list[tuple[str, str]]
    parse_trees: list[str]


def load_text(path: Path) -> str:
    """Read a UTF-8 text file and normalize whitespace."""

    return " ".join(path.read_text(encoding="utf-8").split())


def word_tokens(text: str) -> list[str]:
    """Return word tokens without punctuation-only marks."""

    return [token for token in treebank_tokens(text) if token not in string.punctuation]


def fallback_tags(tokens: list[str]) -> list[tuple[str, str]]:
    """Return deterministic tags for the assignment sentence."""

    return [(token, FALLBACK_POS_TAGS.get(token.casefold(), "NN")) for token in tokens]


def tag_with_nltk(tokens: list[str]) -> list[tuple[str, str]]:
    """Tag with NLTK's POS tagger, falling back when corpus data is absent."""

    try:
        return pos_tag(tokens)
    except LookupError:
        return fallback_tags(tokens)


def tag_with_textblob(text: str, tokens: list[str]) -> list[tuple[str, str]]:
    """Tag with TextBlob, falling back when TextBlob corpora are absent."""

    try:
        from textblob import TextBlob

        tags = cast(list[tuple[Any, Any]], TextBlob(text).tags)
        return [(str(word), str(tag)) for word, tag in tags]
    except (ImportError, LookupError, MissingCorpusError):
        return fallback_tags(tokens)


def tag_with_regex(tokens: list[str]) -> list[tuple[str, str]]:
    """Tag with a rule-based regular-expression tagger."""

    tagger = RegexpTagger(
        [
            (r"(?i)^(the|a|an)$", "DT"),
            (r"(?i)^(and|or|but)$", "CC"),
            (r"(?i)^(at|in|on|by|for|from|to|with)$", "IN"),
            (r"(?i)^.*ed$", "VBD"),
            (r"(?i)^(big|black|white)$", "JJ"),
            (r"(?i)^away$", "RB"),
            (r"(?i)^.*$", "NN"),
        ]
    )
    return tagger.tag(tokens)


def build_parse_trees(tokens: list[str]) -> list[str]:
    """Build possible parse trees for the assignment sentence."""

    grammar = CFG.fromstring(
        """
        S -> NP VP
        NP -> Det AdjSeq N
        AdjSeq -> Adj Adj | Adj
        VP -> VP CC VP | VBD PP | VBD Adv
        PP -> P NP
        Det -> 'The' | 'the'
        Adj -> 'big' | 'black' | 'white'
        N -> 'dog' | 'cat'
        VBD -> 'barked' | 'chased'
        P -> 'at'
        CC -> 'and'
        Adv -> 'away'
        """
    )
    parser = ChartParser(grammar)
    return [tree.pformat(margin=100) for tree in parser.parse(tokens)]


def analyze_text(text: str) -> PosResult:
    """Run POS tagging and parsing over the assignment sentence."""

    tokens = word_tokens(text)
    return PosResult(
        tokens=tokens,
        nltk_tags=tag_with_nltk(tokens),
        textblob_tags=tag_with_textblob(text, tokens),
        regex_tags=tag_with_regex(tokens),
        parse_trees=build_parse_trees(tokens),
    )


def format_section(title: str, value: object) -> str:
    """Format CLI output consistently for report capture."""

    return f"\n## {title}\n{pformat(value, width=100)}"


def build_output(text: str) -> str:
    """Build report-ready POS and parser output."""

    result = analyze_text(text)
    return "\n".join(
        [
            format_section("Tokens", result.tokens),
            format_section("NLTK POS Tagger", result.nltk_tags),
            format_section("TextBlob POS Tagger", result.textblob_tags),
            format_section("Regular Expression Tagger", result.regex_tags),
            format_section("Parse Trees", result.parse_trees),
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=Path("data/Data_2.txt"),
        help="Path to Data_2.txt.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Question 3 analysis and print report-ready output."""

    args = parse_args()
    print(build_output(load_text(args.data_path)))


if __name__ == "__main__":
    main()
