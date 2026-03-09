import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    ans1 = []

    for pos in range(seq_len):
        ans2 = []

        for j in range(d_model):

            if j % 2 == 0:  # even index
                pe = np.sin(pos / (base ** ((2 * (j//2)) / d_model)))
            else:           # odd index
                pe = np.cos(pos / (base ** ((2 * (j//2)) / d_model)))

            ans2.append(pe)

        ans1.append(ans2)

    ans1 = np.array(ans1, dtype=float)

    return ans1