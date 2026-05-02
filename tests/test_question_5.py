from pathlib import Path

from txsa_assignment.question_1 import load_text
from txsa_assignment.question_5 import (
    build_comparisons,
    tokenize_with_treebank,
    tokenize_with_wordpiece,
    wordpiece_tokenize_word,
)


DATA_PATH = Path("data/Data_1.txt")


def test_treebank_tokenizer_matches_group_method() -> None:
    text = load_text(DATA_PATH)
    tokens, note = tokenize_with_treebank(text)

    assert "Treebank" in note
    assert tokens[:5] == ["Classification", "is", "the", "task", "of"]
    assert "." in tokens


def test_wordpiece_splits_known_subword() -> None:
    assert wordpiece_tokenize_word("multiclass", {"multi", "##class"}) == ["multi", "##class"]


def test_wordpiece_tokenizer_is_deterministic() -> None:
    tokens, note = tokenize_with_wordpiece("multiclass classification.")

    assert "WordPiece" in note
    assert tokens == ["multi", "##class", "classification", "."]


def test_build_comparisons_returns_all_report_sections() -> None:
    comparisons = build_comparisons(load_text(DATA_PATH))

    assert [comparison.name for comparison in comparisons] == [
        "SpaCy Tokenizer",
        "TreebankWord Tokenizer",
        "BERT/WordPiece Tokenizer",
        "Stanza Tokenizer",
    ]
    assert all(comparison.tokens for comparison in comparisons)
    assert all(comparison.implementation_note for comparison in comparisons)
