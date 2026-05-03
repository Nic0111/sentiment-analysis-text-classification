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
    """One conditional probability used in a sentence calculation."""

    bigram: tuple[str, str]
    numerator: int
    denominator: int
    probability: Fraction


@dataclass(frozen=True)
class BigramResult:
    """Complete unsmoothed and smoothed bigram analysis result."""

    training_sentences: list[list[str]]
    target_sentence: list[str]
    vocabulary: list[str]
    unsmoothed_steps: list[BigramStep]
    smoothed_steps: list[BigramStep]
    unsmoothed_probability: Fraction
    smoothed_probability: Fraction


def load_text(path: Path) -> str:
    """Read the raw Q4 text file."""

    return path.read_text(encoding="utf-8")


def extract_padded_sentences(text: str) -> list[list[str]]:
    """Extract all padded sentences and keep <s>/</s> as tokens."""

    matches = re.findall(r"<s>\s+(.*?)\s+</s>", text)
    return [["<s>", *match.split(), "</s>"] for match in matches]


def split_training_and_target(text: str) -> tuple[list[list[str]], list[str]]:
    sentences = extract_padded_sentences(text)
    if len(sentences) < 2:
        raise ValueError("Expected at least one training sentence and one target sentence.")

    return sentences[:-1], sentences[-1]


def bigrams(sentence: list[str]) -> list[tuple[str, str]]:
    """Return adjacent token pairs for a sentence."""

    return list(zip(sentence, sentence[1:]))


def count_unigrams(sentences: list[list[str]]) -> Counter[str]:
    """Count each token in the training corpus."""

    return Counter(token for sentence in sentences for token in sentence)


def count_bigrams(sentences: list[list[str]]) -> Counter[tuple[str, str]]:
    """Count each bigram in the training corpus."""

    return Counter(pair for sentence in sentences for pair in bigrams(sentence))


def unsmoothed_bigram_step(
    bigram: tuple[str, str],
    unigram_counts: Counter[str],
    bigram_counts: Counter[tuple[str, str]],
) -> BigramStep:
    """Calculate P(w_i | w_i-1) without smoothing for one bigram."""

    previous_token, current_token = bigram
    numerator = bigram_counts[(previous_token, current_token)]
    denominator = unigram_counts[previous_token]

    return BigramStep(
        bigram=bigram,
        numerator=numerator,
        denominator=denominator,
        probability=Fraction(numerator, denominator),
    )


def smoothed_bigram_step(
    bigram: tuple[str, str],
    unigram_counts: Counter[str],
    bigram_counts: Counter[tuple[str, str]],
    vocabulary_size: int,
) -> BigramStep:
    """Calculate add-one smoothed P(w_i | w_i-1) for one bigram."""

    previous_token, current_token = bigram
    numerator = bigram_counts[(previous_token, current_token)] + 1
    denominator = unigram_counts[previous_token] + vocabulary_size

    return BigramStep(
        bigram=bigram,
        numerator=numerator,
        denominator=denominator,
        probability=Fraction(numerator, denominator),
    )


def multiply(probabilities: list[Fraction]) -> Fraction:
    """Multiply exact probability fractions."""

    result = Fraction(1, 1)
    for probability in probabilities:
        result *= probability
    return result


def sentence_probability(steps: list[BigramStep]) -> Fraction:
    """Calculate a sentence probability from its bigram steps."""

    return multiply([step.probability for step in steps])


def analyze_text(text: str) -> BigramResult:
    """Run unsmoothed and smoothed bigram analysis over the Q4 text."""

    training_sentences, target_sentence = split_training_and_target(text)
    unigram_counts = count_unigrams(training_sentences)
    bigram_counts = count_bigrams(training_sentences)
    vocabulary = sorted(unigram_counts)
    vocabulary_size = len(vocabulary)
    target_bigrams = bigrams(target_sentence)

    unsmoothed_steps = [
        unsmoothed_bigram_step(bigram, unigram_counts, bigram_counts)
        for bigram in target_bigrams
    ]
    smoothed_steps = [
        smoothed_bigram_step(
            bigram=bigram,
            unigram_counts=unigram_counts,
            bigram_counts=bigram_counts,
            vocabulary_size=vocabulary_size,
        )
        for bigram in target_bigrams
    ]

    return BigramResult(
        training_sentences=training_sentences,
        target_sentence=target_sentence,
        vocabulary=vocabulary,
        unsmoothed_steps=unsmoothed_steps,
        smoothed_steps=smoothed_steps,
        unsmoothed_probability=sentence_probability(unsmoothed_steps),
        smoothed_probability=sentence_probability(smoothed_steps),
    )


def fraction_to_float(probability: Fraction) -> float:
    """Convert an exact probability to a readable decimal."""

    return float(probability)


def step_summary(steps: list[BigramStep]) -> list[dict[str, object]]:
    """Return serializable calculation steps for CLI output."""

    return [
        {
            "bigram": step.bigram,
            "Count(w_i-1, w_i)": step.numerator,
            "Count(w_i-1)": step.denominator,
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
