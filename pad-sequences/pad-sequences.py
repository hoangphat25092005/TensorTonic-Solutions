import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    N = len(seqs)

    if max_len is None:
        L = max(len(seq) for seq in seqs) if seqs else 0
    else:
        L = max_len

    padded = []

    for seq in seqs:
        new_seq = seq[:L]  # truncate if longer
        new_seq = new_seq + [pad_value] * (L - len(new_seq))
        padded.append(new_seq)

    return np.array(padded)