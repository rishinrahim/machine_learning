# Word Vectors and Word Senses

## Word Representation

- How we represent words is the first and most important denominator across all NLP tasks
- **Denotational semantics**: The concept of representing an idea as a symbol (a word or a one-hot vector). It is sparse and cannot capture similarity. There is no notion of similarity or order or context This is a "localist" representation.
- **Word order**
    - Order of words is important. The rules that govern the word order in a sequence of words(sentence) is called grammar of a language
    - Word order and  Grammar are important when a statement intends to represent the logical relationship between things
    - This is the Reason why computer languages depend on rigid grammar and syntax rule parsers
- **Distributional Semantics :**
    - Based on distributional hypothesis : linguistic items with similar distributions have similar meanings.
    - A word's meaning is given by the words that frequently appear close by
    - When a Word W appears in a text, its context is the set of words that appear near by ( within a fixed size window)
    - Use the many contexts of W to build up the representation of W

    Word Vectors

    **One-hot vector**: Represent every word as an R|V|×1 vector with all 0s and one 1 at the index of that word in the sorted
    English language.

    **SVD based Methods**

    **Iteration based Methods**