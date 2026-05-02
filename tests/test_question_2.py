from pathlib import Path

from txsa_assignment.question_2 import analyze_text, content_tokens, load_text


DATA_PATH = Path("data/Data_1.txt")


def test_content_tokens_are_ready_for_stemming() -> None:
    tokens = content_tokens(load_text(DATA_PATH))

    assert "Classification" in tokens
    assert "the" not in [token.casefold() for token in tokens]
    assert "." not in tokens


def test_stemming_outputs_are_deterministic() -> None:
    result = analyze_text(load_text(DATA_PATH))

    assert len(result.source_tokens) == len(result.regex_stems)
    assert len(result.source_tokens) == len(result.porter_stems)
    assert len(result.source_tokens) == len(result.lancaster_stems)
    assert result.regex_stems[result.source_tokens.index("choosing")] == "choos"
    assert result.porter_stems[result.source_tokens.index("classification")] == "classif"
    assert result.lancaster_stems[result.source_tokens.index("classification")] == "class"
