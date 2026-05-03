# Part A Question 2: Word Stemming

## 1. Importance of Stemming

Stemming reduces related word forms to a shorter common base form. This is
important in text analytics because words such as `classifies`, `classified`,
and `classification` may refer to similar ideas but appear as separate tokens.
By reducing variants, stemming can improve word-frequency analysis, indexing,
search, and feature extraction for machine-learning models.

Stemming is not the same as lemmatization. It does not always produce a real
dictionary word, but it is fast and useful when the goal is to group similar
surface forms.

## 2. Stemming Output

The tokens were first tokenized with the same NLTK `word_tokenize` approach used
in Q1, then NLTK stop words and punctuation were removed before stemming. This
keeps Q2 preprocessing consistent with the selected Q1 tokenization method.

### Regular Expression Stemmer

The regular expression stemmer removes selected suffixes such as `ing`, `ed`,
`s`, and `ly`. Example output includes:

```text
choosing -> choos
tasks -> task
considered -> consider
labels -> label
interesting -> interest
classified -> classifi
```

### Porter Stemmer

The Porter stemmer applies a standard sequence of stemming rules. Example output
includes:

```text
classification -> classif
considered -> consid
isolation -> isol
advance -> advanc
sequence -> sequenc
jointly -> jointli
```

### Lancaster Stemmer

The Lancaster stemmer is more aggressive than Porter. Example output includes:

```text
classification -> class
given -> giv
basic -> bas
advance -> adv
number -> numb
sequence -> sequ
```

## 3. Differences Between the Stemmers

The regular expression stemmer is the simplest method because it only removes
suffixes that match the manually provided pattern. It is easy to understand, but
it can make rough stems such as `clas` from `class` when the final `s` matches
the rule.

The Porter stemmer is more balanced. It reduces many words effectively while
remaining less aggressive than Lancaster. For example, it changes
`classification` to `classif` and `sequence` to `sequenc`.

The Lancaster stemmer is the most aggressive. It produces shorter stems such as
`classification -> class` and `advance -> adv`. This can group more variants
together, but it may also remove too much information for some analytics tasks.
