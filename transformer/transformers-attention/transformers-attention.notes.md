import torch

import math

def scaled_dot_product_attention(Q, K, V):

    d_k = K.shape[-1]

    # QK^T

    scores = torch.matmul(

        Q,

        K.transpose(-2, -1)

    )

    # Scale

    scores = scores / math.sqrt(d_k)

    # Manual softmax

    exp_scores = torch.exp(scores)

    sum_exp = torch.sum(

        exp_scores,

        dim=-1,

        keepdim=True

    )

    weights = exp_scores / sum_exp

    # Weighted sum

    output = torch.matmul(

        weights,

        V

    )

    return output