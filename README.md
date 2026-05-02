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
```

## Run Tests

```bash
uv run pytest
```
