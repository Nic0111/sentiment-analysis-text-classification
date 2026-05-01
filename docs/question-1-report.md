# Part A Question 1: Tokenization and Filtering

## Data Used

The analysis uses `Data_1.txt`, which describes classification tasks and their
variants in text analytics.

## 1. Word Tokenization

Three tokenization methods were applied to the same text corpus.

### Split Function

The split function separates text only by whitespace. This makes it simple, but
punctuation remains attached to words.

```text
['Classification', 'is', 'the', 'task', 'of', 'choosing', 'the', 'correct',
 'class', 'label', 'for', 'a', 'given', 'input.', 'In', 'basic',
 'classification', 'tasks,', 'each', 'input', 'is', 'considered', 'in',
 'isolation', 'from', 'all', 'other', 'inputs,', 'and', 'the', 'set', 'of',
 'labels', 'is', 'defined', 'in', 'advance.', 'The', 'basic',
 'classification', 'task', 'has', 'a', 'number', 'of', 'interesting',
 'variants.', 'For', 'example,', 'in', 'multiclass', 'classification,', 'each',
 'instance', 'may', 'be', 'assigned', 'multiple', 'labels;', 'in', 'open-class',
 'classification,', 'the', 'set', 'of', 'labels', 'is', 'not', 'defined', 'in',
 'advance;', 'and', 'in', 'sequence', 'classification,', 'a', 'list', 'of',
 'inputs', 'are', 'jointly', 'classified.']
```

### Regular Expression

The regular expression tokenizer separates words and punctuation. It also keeps
hyphenated terms such as `open-class` together.

```text
['Classification', 'is', 'the', 'task', 'of', 'choosing', 'the', 'correct',
 'class', 'label', 'for', 'a', 'given', 'input', '.', 'In', 'basic',
 'classification', 'tasks', ',', 'each', 'input', 'is', 'considered', 'in',
 'isolation', 'from', 'all', 'other', 'inputs', ',', 'and', 'the', 'set', 'of',
 'labels', 'is', 'defined', 'in', 'advance', '.', 'The', 'basic',
 'classification', 'task', 'has', 'a', 'number', 'of', 'interesting',
 'variants', '.', 'For', 'example', ',', 'in', 'multiclass', 'classification',
 ',', 'each', 'instance', 'may', 'be', 'assigned', 'multiple', 'labels', ';',
 'in', 'open-class', 'classification', ',', 'the', 'set', 'of', 'labels', 'is',
 'not', 'defined', 'in', 'advance', ';', 'and', 'in', 'sequence',
 'classification', ',', 'a', 'list', 'of', 'inputs', 'are', 'jointly',
 'classified', '.']
```

### NLTK

NLTK Treebank tokenization gives a linguistically informed tokenization output
without requiring the text to be manually split.

```text
['Classification', 'is', 'the', 'task', 'of', 'choosing', 'the', 'correct',
 'class', 'label', 'for', 'a', 'given', 'input', '.', 'In', 'basic',
 'classification', 'tasks', ',', 'each', 'input', 'is', 'considered', 'in',
 'isolation', 'from', 'all', 'other', 'inputs', ',', 'and', 'the', 'set', 'of',
 'labels', 'is', 'defined', 'in', 'advance', '.', 'The', 'basic',
 'classification', 'task', 'has', 'a', 'number', 'of', 'interesting',
 'variants', '.', 'For', 'example', ',', 'in', 'multiclass', 'classification',
 ',', 'each', 'instance', 'may', 'be', 'assigned', 'multiple', 'labels', ';',
 'in', 'open-class', 'classification', ',', 'the', 'set', 'of', 'labels', 'is',
 'not', 'defined', 'in', 'advance', ';', 'and', 'in', 'sequence',
 'classification', ',', 'a', 'list', 'of', 'inputs', 'are', 'jointly',
 'classified', '.']
```

## 2. Most Suitable Tokenization Operation

NLTK tokenization is the most suitable option for text analytics in this case.
The split function is too basic because tokens such as `input.`, `tasks,`, and
`labels;` keep punctuation attached to the word. This can create duplicate or
misleading terms during later analysis. The regular expression tokenizer improves
on split tokenization by separating punctuation, but NLTK is preferable because
it is a standard NLP tokenizer designed to handle common English tokenization
patterns consistently.

## 3. Stop Words and Punctuation Removal

The stop words found in the text corpus are:

```text
['a', 'all', 'and', 'are', 'be', 'for', 'from', 'has', 'in', 'is', 'may', 'not',
 'of', 'other', 'the']
```

After removing stop words and punctuation from the NLTK tokens, the remaining
tokens are:

```text
['Classification', 'task', 'choosing', 'correct', 'class', 'label', 'given',
 'input', 'basic', 'classification', 'tasks', 'each', 'input', 'considered',
 'isolation', 'inputs', 'set', 'labels', 'defined', 'advance', 'basic',
 'classification', 'task', 'number', 'interesting', 'variants', 'example',
 'multiclass', 'classification', 'each', 'instance', 'assigned',
 'multiple', 'labels', 'open-class', 'classification', 'set', 'labels',
 'defined', 'advance', 'sequence', 'classification', 'list', 'inputs',
 'jointly', 'classified']
```

## 4. Importance of Filtering Stop Words and Punctuation

Filtering stop words and punctuation is important because these tokens usually
carry little topic meaning. Words such as `the`, `is`, `of`, and `in` occur
frequently in English text, but they do not strongly describe the subject of the
corpus. Punctuation is also usually not useful for basic word-frequency analysis.

After filtering, the remaining tokens focus more clearly on meaningful terms
such as `Classification`, `task`, `label`, `input`, `labels`, and `sequence`.
This improves later text analytics tasks by reducing noise and making the
important content words easier to compare, count, and interpret.
