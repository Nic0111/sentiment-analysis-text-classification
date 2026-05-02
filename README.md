# TXSA Assignment

This repository contains runnable Python work for the CT107-3-3 Text Analytics
and Sentiment Analysis group assignment.

The project follows a small harness-engineering style: keep the assignment
context, commands, data, code, tests, and report notes inside the repository so
future runs are easy to inspect and repeat.

## Project Map

- `data/Data_1.txt` - source text for Part A Q1 and Q2.
- `data/Data_2.txt` - source sentence for Part A Q3.
- `data/Data_3.txt` - training corpus and target sentence for Part A Q4.
- `src/txsa_assignment/question_*.py` - Part A implementations.
- `tests/test_question_*.py` - focused checks for the implemented behavior.
- `docs/question-*-report.md` - report-ready explanations and output summaries.

## Setup

Install dependencies with `uv`:

```bash
uv sync --extra dev
```

## spaCy And Stanza Setup

Q5 requires the real spaCy and Stanza tokenizers. Install the packages and
download their English resources before running Q5:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -c "import stanza; stanza.download('en', processors='tokenize')"
```

Quick checks:

```bash
uv run python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print([t.text for t in nlp('This is a test.')])"
uv run python -c "import stanza; nlp = stanza.Pipeline('en', processors='tokenize', download_method=None, verbose=False); print([t.text for s in nlp('This is a test.').sentences for t in s.tokens])"
```

Official references: spaCy model downloads are handled through
`python -m spacy download en_core_web_sm`, and Stanza models are downloaded with
`stanza.download(...)`.

## Run Part A Q1

```bash
uv run python -m txsa_assignment.question_1 data/Data_1.txt
```

Or use the script entry point:

```bash
uv run txsa-q1 data/Data_1.txt
```

## Run The Remaining Part A Questions

```bash
uv run python -m txsa_assignment.question_2 data/Data_1.txt
uv run python -m txsa_assignment.question_3 data/Data_2.txt
uv run python -m txsa_assignment.question_4 data/Data_3.txt
uv run python -m txsa_assignment.question_5 data/Data_1.txt
```

## Run Tests

```bash
uv run pytest
```
