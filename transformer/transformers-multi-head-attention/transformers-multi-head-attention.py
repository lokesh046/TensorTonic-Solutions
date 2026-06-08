import numpy as np
import math


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    batch_size,seq_len,d_model = Q.shape

    d_k = d_model // num_heads


    Q_proj = np.matmul(Q,W_q)
    K_proj = np.matmul(K,W_k)
    V_proj = np.matmul(V,W_v)

    Q_proj = Q_proj.reshape(
        batch_size,
        seq_len,
        num_heads,
        d_k
    )

    K_proj = K_proj.reshape(
        batch_size,
        seq_len,
        num_heads,
        d_k
    )

    V_proj = V_proj.reshape(
        batch_size,
        seq_len,
        num_heads,
        d_k
    )

    Q_proj = Q_proj.transpose(0,2,1,3)
    K_proj = K_proj.transpose(0,2,1,3)
    V_proj = V_proj.transpose(0,2,1,3)


    scores = np.matmul(
        Q_proj,
        K_proj.transpose(0,1,3,2)
    )

    scores = scores / math.sqrt(d_k)


    attention_weight = softmax(scores)

    head_outputs = np.matmul(attention_weight,V_proj)

    head_outputs = head_outputs.transpose(
        0,
        2,
        1,
        3
    )

    concat  = head_outputs.reshape(
        batch_size,
        seq_len,
        d_model
    )

    output = np.matmul(
        concat,
        W_o
    )
    return output

    
    
    