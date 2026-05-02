from fractions import Fraction
from pathlib import Path

from txsa_assignment.question_4 import analyze_text, bigrams, load_text, split_training_and_target


DATA_PATH = Path("data/Data_3.txt")


def test_split_training_and_target() -> None:
    training, target = split_training_and_target(load_text(DATA_PATH))

    assert len(training) == 3
    assert target == ["<s>", "I", "read", "a", "different", "book", "by", "Danielle", "</s>"]


def test_bigrams_returns_adjacent_pairs() -> None:
    assert bigrams(["<s>", "I", "read"]) == [("<s>", "I"), ("I", "read")]


def test_bigram_probabilities_are_exact() -> None:
    result = analyze_text(load_text(DATA_PATH))

    assert result.unsmoothed_probability == Fraction(1, 27)
    assert result.smoothed_probability == Fraction(512, 418161601)
    assert result.vocabulary == ["</s>", "<s>", "Danielle", "He", "I", "a", "book", "by", "different", "read"]
