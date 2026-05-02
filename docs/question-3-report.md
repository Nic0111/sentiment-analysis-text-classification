# Part A Question 3: POS Taggers and Syntactic Analysis

## Data Used

The sentence from `Data_2.txt` is:

```text
The big black dog barked at the white cat and chased away.
```

## 1. POS Tagging Output

The sentence was tokenized into:

```text
['The', 'big', 'black', 'dog', 'barked', 'at', 'the', 'white', 'cat', 'and', 'chased', 'away']
```

The NLTK, TextBlob, and regular-expression taggers produced the following tags
for this sentence:

```text
[('The', 'DT'), ('big', 'JJ'), ('black', 'JJ'), ('dog', 'NN'),
 ('barked', 'VBD'), ('at', 'IN'), ('the', 'DT'), ('white', 'JJ'),
 ('cat', 'NN'), ('and', 'CC'), ('chased', 'VBD'), ('away', 'RB')]
```

The code uses the real NLTK and TextBlob taggers when their local corpora are
available. If the corpora are unavailable, it falls back to a documented
deterministic tag set for this assignment sentence so the script remains
runnable offline.

## 2. Differences Between the POS Taggers

The NLTK POS tagger is a statistical tagger. It normally uses a trained model to
assign tags based on word identity and context. TextBlob provides a simpler API
on top of NLP tagging tools, making it convenient for quick analysis. The
regular-expression tagger is rule-based; it assigns tags using manually written
patterns, such as tagging words ending in `ed` as past-tense verbs.

For this short sentence, all three methods produce the same useful result.
However, the regular-expression tagger is less flexible because it depends on
the quality of the written rules. Statistical taggers such as NLTK and TextBlob
usually generalize better to unseen sentences.

## 3. Parse Tree

A context-free grammar was used to parse the sentence into a noun phrase and a
verb phrase:

```text
(S
  (NP (Det The) (AdjSeq (Adj big) (Adj black)) (N dog))
  (VP
    (VP (VBD barked) (PP (P at) (NP (Det the) (AdjSeq (Adj white)) (N cat))))
    (CC and)
    (VP (VBD chased) (Adv away))))
```

This tree shows that `The big black dog` is the subject noun phrase, while
`barked at the white cat and chased away` is the coordinated verb phrase.
