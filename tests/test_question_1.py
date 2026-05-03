from pathlib import Path

from txsa_assignment.question_1 import (
    FilteringResult,
    get_stopwords,
    load_text,
    remove_stopwords_and_punctuation,
    tokenize_with_nltk,
    tokenize_with_regex,
    tokenize_with_split,
)


DATA_PATH = Path("data/Data_1.txt")


def test_load_text_normalizes_line_breaks() -> None:
    text = load_text(DATA_PATH)

    assert "\n" not in text
    assert text.startswith("Classification is the task")


def test_split_tokenization_is_deterministic() -> None:
    tokens = tokenize_with_split(load_text(DATA_PATH))

    assert tokens[:5] == ["Classification", "is", "the", "task", "of"]
    assert "input." in tokens
    assert len(tokens) == 82


def test_regex_tokenization_separates_punctuation() -> None:
    tokens = tokenize_with_regex(load_text(DATA_PATH))

    assert tokens[:6] == ["Classification", "is", "the", "task", "of", "choosing"]
    assert "." in tokens
    assert ["open", "-", "class"] == tokens[tokens.index("open") : tokens.index("open") + 3]
    assert len(tokens) == 96


def test_nltk_tokenization_is_deterministic() -> None:
    tokens = tokenize_with_nltk(load_text(DATA_PATH))

    assert tokens[:6] == ["Classification", "is", "the", "task", "of", "choosing"]
    assert "." in tokens
    assert "open-class" in tokens
    assert len(tokens) == 94


def test_stopwords_are_deterministic() -> None:
    stopwords = get_stopwords()

    assert "may" in stopwords
    assert "each" not in stopwords


def test_filtering_removes_stopwords_case_insensitively_and_punctuation() -> None:
    result = remove_stopwords_and_punctuation(["The", "Classification", "is", "."])

    assert result == FilteringResult(
        filtered_tokens=["Classification"],
        stopwords_found=["is", "the"],
    )


def test_filtering_result_includes_stopwords_and_filtered_tokens() -> None:
    tokens = tokenize_with_nltk(load_text(DATA_PATH))
    result = remove_stopwords_and_punctuation(tokens)

    assert isinstance(result.filtered_tokens, list)
    assert isinstance(result.stopwords_found, list)
    assert "Classification" in result.filtered_tokens
    assert "." not in result.filtered_tokens
    assert "the" in result.stopwords_found
    assert "may" in result.stopwords_found
    assert "each" not in result.stopwords_found
