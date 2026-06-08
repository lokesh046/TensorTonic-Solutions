import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = K.shape[-1]

    scores = torch.matmul(
        Q,K.transpose(-2,-1)
    )

    scores = scores / math.sqrt(d_k)

    exp_scores = torch.exp(scores)

    sum_exp = torch.sum(exp_scores,dim=-1,keepdim = True)

    weight = exp_scores /sum_exp

    output = torch.matmul(weight,
                         V)
    return output

    

    

    