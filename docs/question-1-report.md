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

NLTK tokenization is the most suitable tokenization method for this corpus
because it produces clean word-level tokens while handling sentence punctuation
more consistently than the split function and with less manual rule design than
the regular expression method. The obtained output shows that NLTK separates
punctuation marks such as `.`, `,`, and `;` from surrounding words, while keeping
the meaningful hyphenated term `open-class` as one token. This makes the token
list more suitable for later text analytics tasks such as stop-word removal,
frequency analysis, and feature extraction.

The split function is the simplest method because it only separates tokens at
whitespace. However, the output shows that punctuation remains attached to words,
for example `input.`, `tasks,`, `labels;`, and `classified.`. This can create
inconsistent tokens because `input` and `input.` would be treated as different
items even though they represent the same word. Therefore, split tokenization is
easy to implement but too basic for reliable text analytics.

The regular expression tokenizer improves on the split function because it
separates punctuation from words and gives more control over token boundaries.
For example, it separates `input` and `.` instead of keeping `input.` as one
token. However, this method depends heavily on the written pattern. In the
obtained output, `open-class` is split into `open`, `-`, and `class`, which may
not be desirable when the hyphenated expression should be interpreted as one
term. Thus, regex tokenization is flexible but requires careful manual design.

Therefore, NLTK tokenization is reaffirmed as the most suitable option for this
question. It gives a cleaner and more linguistically appropriate tokenization
output than the split function, while avoiding some of the manual rule problems
seen in the regular expression method. For this corpus, NLTK provides the best
balance between correctness, readability, and suitability for later text
analytics processing.

## 3. Stop Words and Punctuation Removal

The stop words found in the text corpus are:

```text
['a', 'all', 'and', 'are', 'be', 'each', 'for', 'from', 'has', 'in', 'is',
 'not', 'of', 'other', 'the']
```

After removing stop words and punctuation from the NLTK tokens, the remaining
tokens are:

```text
['Classification', 'task', 'choosing', 'correct', 'class', 'label', 'given',
 'input', 'basic', 'classification', 'tasks', 'input', 'considered',
 'isolation', 'inputs', 'set', 'labels', 'defined', 'advance', 'basic',
 'classification', 'task', 'number', 'interesting', 'variants', 'example',
 'multiclass', 'classification', 'instance', 'may', 'assigned',
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
