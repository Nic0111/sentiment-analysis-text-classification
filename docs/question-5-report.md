# Part A Question 5: Alternative Tokenization Approaches

The group tokenization method used for comparison is the NLTK Treebank tokenizer
from Q1. The following sections compare four alternative tokenizer styles using
the same `Data_1.txt` corpus.

## SpaCy Tokenizer

### 1. Alternative Tokenization Method

The spaCy tokenizer is used through the installed `en_core_web_sm` English
pipeline. It is the first processing stage in a spaCy NLP pipeline, before later
steps such as POS tagging, dependency parsing, and named entity recognition.

### 2. Comparison With Group Tokenization Method

For this corpus, spaCy tokenization is very similar to the group Treebank method
because both separate punctuation such as `.`, `,`, and `;` from word tokens.
One visible difference is that spaCy splits `open-class` into `open`, `-`, and
`class`, while Treebank keeps `open-class` as one token.

The main difference is ecosystem fit. Treebank tokenization is lightweight and
direct, while spaCy tokenization is normally used as part of a wider NLP
pipeline that can continue into POS tagging, dependency parsing, and named
entity recognition.

### 3. Evaluation of Alternative Tokenization Method

spaCy is better when the project needs a full NLP workflow after tokenization.
It is less necessary for this small Q1 corpus because the task only requires
word tokenization and punctuation handling. For basic text analytics, Treebank
is simpler and easier to reproduce.

## TreebankWord Tokenizer

### 1. Alternative Tokenization Method

TreebankWord tokenization follows conventions from the Penn Treebank. It splits
many punctuation marks into separate tokens and handles common English token
patterns consistently.

### 2. Comparison With Group Tokenization Method

This is the same tokenizer used by the group NLTK method in Q1. Therefore, its
output should match the group tokenization output exactly.

### 3. Evaluation of Alternative Tokenization Method

TreebankWord tokenization is a strong choice for this assignment. It is
lightweight, readable, and does not require downloaded model files. It is not as
feature-rich as spaCy or Stanza, but it is highly suitable for demonstrating
tokenization and filtering on `Data_1.txt`.

## BERT/WordPiece Tokenizer

### 1. Alternative Tokenization Method

BERT uses WordPiece tokenization, which breaks text into subword units. Common
words may remain as full tokens, while less common words can be split into
smaller pieces. For example:

```text
multiclass -> multi, ##class
```

The local implementation uses a small assignment-specific WordPiece vocabulary
and longest-match splitting so it can run without downloading a pretrained BERT
vocabulary.

### 2. Comparison With Group Tokenization Method

The group Treebank method produces human-readable word tokens. WordPiece can
produce subword tokens such as `multi` and `##class`, so its output may contain
more tokens and may be less readable for manual analysis.

### 3. Evaluation of Alternative Tokenization Method

WordPiece is better when the output will be used by transformer models such as
BERT. It is worse for simple word-frequency reporting because subword tokens do
not always correspond to complete words. Therefore, it is different rather than
universally better or worse: it is model-oriented instead of report-oriented.

## Stanza Tokenizer

### 1. Alternative Tokenization Method

Stanza is a Stanford NLP toolkit that provides neural NLP pipelines, including
tokenization. The implementation uses the installed English Stanza
`tokenize` processor.

### 2. Comparison With Group Tokenization Method

Stanza and Treebank both aim to produce linguistically sensible tokens. For this
short English paragraph, the resulting word and punctuation boundaries are
similar. Like spaCy, Stanza splits `open-class` into `open`, `-`, and `class`,
while Treebank keeps it as one hyphenated token.

The main difference is resource cost. Treebank is lightweight, while Stanza is
designed for a larger neural NLP workflow and usually requires model resources.

### 3. Evaluation of Alternative Tokenization Method

Stanza is useful when tokenization is only one step in a deeper linguistic
analysis. For Q1-style tokenization, it is heavier than necessary. Treebank is
the more practical method for this assignment unless later tasks need Stanza's
pipeline features.
