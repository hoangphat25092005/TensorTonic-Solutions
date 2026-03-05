import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    if vocab is None or len(vocab) == 0:
        return np.array([], dtype=int)
    vector = [0] * len(vocab)
    for token in tokens:
        if token in vocab:
          vector[vocab.index(token)] += 1
    return np.array(vector)