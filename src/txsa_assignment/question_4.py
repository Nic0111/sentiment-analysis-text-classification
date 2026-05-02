from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from pprint import pformat


@dataclass(frozen=True)
class BigramStep:
    bigram: tuple[str, str]
    numerator: int
    denominator: int
    probability: Fraction


@dataclass(frozen=True)
class BigramResult:
    training_sentences: list[list[str]]
    target_sentence: list[str]
    vocabulary: list[str]
    unsmoothed_steps: list[BigramStep]
    smoothed_steps: list[BigramStep]
    unsmoothed_probability: Fraction
    smoothed_probability: Fraction


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_padded_sentences(text: str) -> list[list[str]]:
    matches = re.findall(r"<s>\s+(.*?)\s+</s>", text)
    return [["<s>", *match.split(), "</s>"] for match in matches]


def split_training_and_target(text: str) -> tuple[list[list[str]], list[str]]:
    sentences = extract_padded_sentences(text)
    if len(sentences) < 2:
        raise ValueError("Expected at least one training sentence and one target sentence.")

    return sentences[:-1], sentences[-1]


def bigrams(sentence: list[str]) -> list[tuple[str, str]]:
    return list(zip(sentence, sentence[1:]))


def multiply(probabilities: list[Fraction]) -> Fraction:
    result = Fraction(1, 1)
    for probability in probabilities:
        result *= probability
    return result


def analyze_text(text: str) -> BigramResult:
    training_sentences, target_sentence = split_training_and_target(text)
    unigram_counts = Counter(token for sentence in training_sentences for token in sentence)
    bigram_counts = Counter(pair for sentence in training_sentences for pair in bigrams(sentence))
    vocabulary = sorted(unigram_counts)
    vocabulary_size = len(vocabulary)

    unsmoothed_steps: list[BigramStep] = []
    smoothed_steps: list[BigramStep] = []

    for previous_token, token in bigrams(target_sentence):
        bigram_count = bigram_counts[(previous_token, token)]
        previous_count = unigram_counts[previous_token]

        unsmoothed_steps.append(
            BigramStep(
                bigram=(previous_token, token),
                numerator=bigram_count,
                denominator=previous_count,
                probability=Fraction(bigram_count, previous_count),
            )
        )
        smoothed_steps.append(
            BigramStep(
                bigram=(previous_token, token),
                numerator=bigram_count + 1,
                denominator=previous_count + vocabulary_size,
                probability=Fraction(bigram_count + 1, previous_count + vocabulary_size),
            )
        )

    return BigramResult(
        training_sentences=training_sentences,
        target_sentence=target_sentence,
        vocabulary=vocabulary,
        unsmoothed_steps=unsmoothed_steps,
        smoothed_steps=smoothed_steps,
        unsmoothed_probability=multiply([step.probability for step in unsmoothed_steps]),
        smoothed_probability=multiply([step.probability for step in smoothed_steps]),
    )


def fraction_to_float(probability: Fraction) -> float:
    """Convert an exact probability to a readable decimal."""

    return float(probability)


def step_summary(steps: list[BigramStep]) -> list[dict[str, object]]:
    """Return serializable calculation steps for CLI output."""

    return [
        {
            "bigram": step.bigram,
            "count": step.numerator,
            "context_count": step.denominator,
            "probability": f"{step.probability.numerator}/{step.probability.denominator}",
        }
        for step in steps
    ]


def build_output(text: str) -> str:
    """Build report-ready language-model output."""

    result = analyze_text(text)
    return "\n".join(
        [
            "\n## Training Sentences\n" + pformat(result.training_sentences, width=100),
            "\n## Target Sentence\n" + pformat(result.target_sentence, width=100),
            "\n## Vocabulary\n" + pformat(result.vocabulary, width=100),
            "\n## Unsmoothed Bigram Steps\n" + pformat(step_summary(result.unsmoothed_steps), width=100),
            "\n## Smoothed Bigram Steps\n" + pformat(step_summary(result.smoothed_steps), width=100),
            (
                "\n## Sentence Probabilities\n"
                f"Unsmoothed: {result.unsmoothed_probability} "
                f"({fraction_to_float(result.unsmoothed_probability):.12f})\n"
                f"Smoothed: {result.smoothed_probability} "
                f"({fraction_to_float(result.smoothed_probability):.12f})"
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=Path("data/Data_3.txt"),
        help="Path to Data_3.txt.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Question 4 analysis and print report-ready output."""

    args = parse_args()
    print(build_output(load_text(args.data_path)))


if __name__ == "__main__":
    main()
