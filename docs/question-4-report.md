# Part A Question 4: Sentence Probabilities

## Data Used

Training corpus:

```text
<s> He read a book </s>
<s> I read a different book </s>
<s> He read a book by Danielle </s>
```

Target sentence:

```text
<s> I read a different book by Danielle </s>
```

## 1. Pre-processing

The sentence pads `<s>` and `</s>` were preserved because they are required for
bigram language modelling. Each sentence was split on whitespace.

The vocabulary size from the training corpus is 10:

```text
['</s>', '<s>', 'Danielle', 'He', 'I', 'a', 'book', 'by', 'different', 'read']
```

The notebook also presents the preprocessed sentences in a table, followed by
separate unigram and bigram count tables. These tables make the probability
calculation traceable because every conditional probability is derived from the
training-corpus counts.

## 2. Manual Unsmoothed Bigram Probability

Using the unsmoothed bigram equation:

```text
P(w_n | w_{n-1}) = Count(w_{n-1}, w_n) / Count(w_{n-1})
```

The target sentence probability is:

```text
P = P(I|<s>) x P(read|I) x P(a|read) x P(different|a)
    x P(book|different) x P(by|book) x P(Danielle|by) x P(</s>|Danielle)

P = 1/3 x 1/1 x 3/3 x 1/3 x 1/1 x 1/3 x 1/1 x 1/1
P = 1/27
P = 0.037037037037
```

## 3. Manual Smoothed Bigram Probability

Using add-one smoothing:

```text
P(w_n | w_{n-1}) = (Count(w_{n-1}, w_n) + 1) / (Count(w_{n-1}) + V)
```

where `V = 10`.

```text
P = 2/13 x 2/11 x 4/13 x 2/13 x 2/11 x 2/13 x 2/11 x 2/11
P = 512/418161601
P = 0.000001224407
```

## 4. Python Implementation Result

The Python implementation reports the same probabilities:

```text
Unsmoothed: 1/27 (0.037037037037)
Smoothed: 512/418161601 (0.000001224407)
```

The smoothed probability is lower because add-one smoothing reserves probability
mass for unseen bigrams in the vocabulary. This prevents unseen bigrams from
receiving zero probability, but it also reduces the probability assigned to
observed bigrams.
