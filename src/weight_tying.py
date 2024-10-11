import torch 
from models import TransformerModelStackTogether, TransformerModel

def tie_head_weights_qk(model):
    '''
        Do weight tying for model with 1 layer, 2 heads for Q, K
    '''
    W = model._backbone.h[0].attn.c_attn.weight
    b = model._backbone.h[0].attn.c_attn.bias

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    W_reshaped = W.view(hidden_size, 3, num_heads, head_dim)
    b_reshaped = b.view(3, num_heads, head_dim)

    with torch.no_grad():
        for i in range(2):  # For Q, K
            W_i = W_reshaped[:, i, :, :]
            b_i = b_reshaped[i, :, :]
            W_i[:, 1, :] = -W_i[:, 0, :]
            b_i[1, :] = -b_i[0, :]