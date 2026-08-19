# MHA 多头注意力的pytorch实现一下，练一下手

import torch
import nn.function as F

def stand_attention(Q, K, V, sm_scale, mask=None):
    '''
    Q (batch_size, num_heads, seq_len, head_dim)
    K (batch_size, num_heads, seq_len, head_dim)
    V (batch_size, num_heads, seq_len, head_dim)
    sm_scale Softmax 缩放引资
    mask， 掩码
    '''

    # (batch_size, num_heads, seq_length, seq_length)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale


    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(attn_scores, dim=-1)
    # (batch_size, num_heads, seq_length, head_dim)
    out = torch.matmul(attn_weights, V)