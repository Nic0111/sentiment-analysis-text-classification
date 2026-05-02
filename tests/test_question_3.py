from pathlib import Path

from txsa_assignment.question_3 import analyze_text, load_text, tag_with_regex, word_tokens


DATA_PATH = Path("data/Data_2.txt")


def test_word_tokens_remove_punctuation() -> None:
    tokens = word_tokens(load_text(DATA_PATH))

    assert tokens == [
        "The",
        "big",
        "black",
        "dog",
        "barked",
        "at",
        "the",
        "white",
        "cat",
        "and",
        "chased",
        "away",
    ]


def test_regex_tagger_assigns_expected_tags() -> None:
    assert tag_with_regex(["The", "dog", "barked"]) == [
        ("The", "DT"),
        ("dog", "NN"),
        ("barked", "VBD"),
    ]


def test_question_3_builds_parse_tree() -> None:
    result = analyze_text(load_text(DATA_PATH))

    assert result.nltk_tags[0] == ("The", "DT")
    assert result.textblob_tags[-1] == ("away", "RB")
    assert result.regex_tags[4] == ("barked", "VBD")
    assert len(result.parse_trees) == 1
    assert "(S" in result.parse_trees[0]
