# TXSA Assignment

This repository contains runnable Python work for the CT107-3-3 Text Analytics
and Sentiment Analysis group assignment.

The project follows a small harness-engineering style: keep the assignment
context, commands, data, code, tests, and report notes inside the repository so
future runs are easy to inspect and repeat.

## Project Map

- `data/Data_1.txt` - source text for Part A Q1 and Q2.
- `src/txsa_assignment/question_1.py` - Part A Q1 implementation.
- `tests/test_question_1.py` - focused checks for the tokenization and filtering
  behavior.
- `docs/question-1-report.md` - report-ready explanation and output summary for
  Part A Q1.

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

## Run Tests

```bash
uv run pytest
```
