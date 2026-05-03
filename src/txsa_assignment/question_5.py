"""Question 5: alternative tokenization approaches for ``Data_1.txt``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import cast

from txsa_assignment.question_1 import load_text, tokenize_with_nltk


@dataclass(frozen=True)
class TokenizerComparison:
    """Tokenizer output and the implementation used to produce it."""

    name: str
    tokens: list[str]
    implementation_note: str


def tokenize_with_spacy(text: str) -> tuple[list[str], str]:
    """Tokenize with the installed spaCy English pipeline."""

    import spacy

    nlp = spacy.load("en_core_web_sm")
    return [token.text for token in nlp(text)], "spaCy en_core_web_sm tokenizer"


def tokenize_with_treebank(text: str) -> tuple[list[str], str]:
    """Tokenize with the same Treebank tokenizer used in the group Q1 method."""

    return tokenize_with_nltk(text), "NLTK TreebankWordTokenizer"


def tokenize_with_wordpiece(text: str) -> tuple[list[str], str]:
    """Tokenize with BERT's pretrained WordPiece tokenizer."""

    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer.tokenize(text), "BERT bert-base-uncased WordPiece tokenizer"


def tokenize_with_stanza(text: str) -> tuple[list[str], str]:
    """Tokenize with the installed Stanza English tokenize processor."""

    import stanza
    from stanza.models.common.doc import Document
    from stanza.pipeline.core import DownloadMethod

    pipeline = stanza.Pipeline(
        lang="en",
        processors="tokenize",
        tokenize_no_ssplit=True,
        download_method=DownloadMethod.NONE,
        verbose=False,
    )
    document = cast(Document, pipeline(text))
    return [token.text for sentence in document.sentences for token in sentence.tokens], "Stanza English tokenizer"


def build_comparisons(text: str) -> list[TokenizerComparison]:
    """Run all Q5 tokenizers and return their outputs."""

    spacy_tokens, spacy_note = tokenize_with_spacy(text)
    treebank_tokens, treebank_note = tokenize_with_treebank(text)
    wordpiece_tokens, wordpiece_note = tokenize_with_wordpiece(text)
    stanza_tokens, stanza_note = tokenize_with_stanza(text)

    return [
        TokenizerComparison(
            name="SpaCy Tokenizer",
            tokens=spacy_tokens,
            implementation_note=spacy_note,
        ),
        TokenizerComparison(
            name="TreebankWord Tokenizer",
            tokens=treebank_tokens,
            implementation_note=treebank_note,
        ),
        TokenizerComparison(
            name="BERT/WordPiece Tokenizer",
            tokens=wordpiece_tokens,
            implementation_note=wordpiece_note,
        ),
        TokenizerComparison(
            name="Stanza Tokenizer",
            tokens=stanza_tokens,
            implementation_note=stanza_note,
        ),
    ]


def format_comparison(comparison: TokenizerComparison) -> str:
    """Format one tokenizer comparison for CLI output."""

    return "\n".join(
        [
            f"\n## {comparison.name}",
            f"Implementation: {comparison.implementation_note}",
            f"Token count: {len(comparison.tokens)}",
            "Tokens:",
            pformat(comparison.tokens, width=100),
        ]
    )


def build_output(text: str) -> str:
    """Build the full Question 5 tokenizer comparison output."""

    return "\n".join(format_comparison(comparison) for comparison in build_comparisons(text))


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
    """Run the Question 5 analysis and print report-ready output."""

    args = parse_args()
    print(build_output(load_text(args.data_path)))


if __name__ == "__main__":
    main()
