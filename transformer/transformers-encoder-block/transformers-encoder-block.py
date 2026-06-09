import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here

    mean = np.mean(x,axis=-1, keepdims=True)
    variance = np.var(x,axis=-1, keepdims=True)

    x_hat = (x - mean) / np.sqrt(variance + eps)

    return gamma * x_hat + beta
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch_size, seq_len, d_model = Q.shape

    d_k = d_model // num_heads

    q_proj = np.matmul(Q,W_q)
    k_proj = np.matmul(K,W_k)
    v_proj = np.matmul(V,W_v)

    q_proj = q_proj.reshape(
        batch_size,
        seq_len,
        num_heads,
        d_k
    ).transpose(0,2,1,3)
    
    k_proj = k_proj.reshape(
        batch_size,
        seq_len,
        num_heads,
        d_k
    ).transpose(0,2,1,3)

    v_proj = v_proj.reshape(
        batch_size,
        seq_len,
        num_heads,
        d_k
    ).transpose(0,2,1,3)

    scores = np.matmul(q_proj,k_proj.transpose(0,1,3,2))

    scores = scores / np.sqrt(d_k)

    scores =scores - np.max(
        scores,
        axis=-1,
        keepdims=True
    )

    exp_scores = np.exp(scores)

    attention_weight = (exp_scores/np.sum(exp_scores,axis=-1,keepdims=True))


    head_output = np.matmul(attention_weight,
                           v_proj)

    head_output = head_output.transpose(
        0,
        2,
        1,
        3)

    concat = head_output.reshape(
        batch_size,
        seq_len,
        d_model
    )

    output = np.matmul(concat,W_o)


    return output


    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = np.matmul(x,W1) +b1

    relu_out = np.maximum(0,hidden)

    output = np.matmul(relu_out,W2) + b2

    return output
def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    attn_output = multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)

    x_prime = layer_norm(x +  attn_output,
                        gamma1,
                        beta1
                        )

    ffn_output = feed_forward(
        x_prime,
        W1,
        b1,
        W2,
        b2
    )

    output = layer_norm(x_prime +  ffn_output,
                       gamma2,beta2)


    return output