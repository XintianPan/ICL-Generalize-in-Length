from models import TransformerModelStackTogether
import os
import wandb
import matplotlib.pyplot as plt
import torch 
from eval import get_run_metrics, read_run_dir, get_model_from_run
import seaborn as sns

from args_parser import get_model_parser

def extract_ou(model):
    '''
    Extract W_O, W_U from model
    return tuple (W_O, W_I)
    '''
    W_u = model._read_out.weight.detach().cpu()
    model = model._backbone
    layer = model.h[0]
    attention = layer.attn

    W_o = attention.c_proj.weight.detach().cpu() 
    return W_o, W_u


def extract_ov_matrices(model):
    # Extract the embedding matrix
    
    W_e = model._read_in.weight.detach().cpu() # shape (hidden_size, input_dim) input_dim = n_dims + 1
    
    
    W_u = model._read_out.weight.detach().cpu()

    # Access the only transformer block
    model = model._backbone
    layer = model.h[0]
    attention = layer.attn

    # Get weights and biases
    c_attn_weight = attention.c_attn.weight.detach().cpu()
    c_attn_bias = attention.c_attn.bias.detach().cpu()

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    

    # Split weights and biases
    W_q = c_attn_weight[:, :hidden_size]
    W_k = c_attn_weight[:, hidden_size:2*hidden_size]
    W_v = c_attn_weight[:, 2*hidden_size:]

    b_q = c_attn_bias[:hidden_size]
    b_k = c_attn_bias[hidden_size:2*hidden_size]
    b_v = c_attn_bias[2*hidden_size:]

    W_o = attention.c_proj.weight.detach().cpu() 

    ov_matrices = []

    ov_matrices.append(
        {
            'W_q': W_q.clone(),
            'W_k': W_k.clone(),
            'W_v': W_v.clone(),
            'b_q': b_q.clone(),
            'b_k': b_k.clone(),
            'b_v': b_v.clone(),
            'W_e': W_e.clone(),
            'W_o': W_o.clone(),
            'W_u': W_u,
        }
    )
    W_o = W_o.transpose(-2, -1)

    W_q = W_q.view(hidden_size, num_heads, head_dim)
    W_k = W_k.view(hidden_size, num_heads, head_dim)
    W_v = W_v.view(hidden_size, num_heads, head_dim)
    W_o = W_o.view(hidden_size, num_heads, head_dim)


    b_q = b_q.view(num_heads, head_dim)
    b_k = b_k.view(num_heads, head_dim)
    b_v = b_v.view(num_heads, head_dim)

    # Collect matrices for all heads

    for head_idx in range(num_heads):
        W_q_head = W_q[:, head_idx, :]  # Shape: (hidden_size, head_dim)
        W_k_head = W_k[:, head_idx, :]
        W_v_head = W_v[:, head_idx, :]

        b_q_head = b_q[head_idx, :]     # Shape: (head_dim,)
        b_k_head = b_k[head_idx, :]
        b_v_head = b_v[head_idx, :]

        W_o_head = W_o[:, head_idx, :]

        W_o_head = W_o_head.transpose(-2, -1)


        ov_matrices.append({
            'W_q': W_q_head,
            'W_k': W_k_head,
            'W_v': W_v_head,
            'b_q': b_q_head,
            'b_k': b_k_head,
            'b_v': b_v_head,
            'W_e': W_e,
            'W_o': W_o_head,
            'W_u': W_u,
        })
    


    return ov_matrices

def heatmap_draw_ov(ov_matrices, title):
    W_v = ov_matrices['W_v']
    W_o = ov_matrices['W_o']
    OV_matrix = W_v @ W_o
    my_array = OV_matrix.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(my_array, cmap='viridis')
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def heatmap_draw_ov_ebmeds(ov_matrices, title):
    # We want to visualize QK ciruit
    # Then output matrix is W_e^T * W_v * W_o * W_u^T
    W_e = ov_matrices['W_e']
    W_v = ov_matrices['W_v']
    W_o = ov_matrices['W_o']
    W_u = ov_matrices['W_u']

    W_e_T = W_e.transpose(0, 1)
    W_u_T = W_u.transpose(0, 1)


    ov_circuit = W_e_T @ W_v @ W_o @ W_u_T

    my_array = ov_circuit.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(my_array, cmap='viridis', annot=True)
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def visualize_ov_from_data(run_path, step=-1):
    model, conf = get_model_from_run(run_path, step)
    wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project + "-vis-both",
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=conf.wandb.name,
            resume=True,
    )
    model = model.cuda().eval()

    ov_matrices = extract_ov_matrices(model)
    for i, ov in enumerate(ov_matrices):
        if i == 0:
            title = "OV circuit"
        else:
            title = f"OV circuit for head {i - 1}"

        heatmap_draw_ov_ebmeds(ov, title)
    # title = "OV circuit"
    
    # title = "OV matrix"
    # heatmap_draw_ov(ov_matrices, title)

    wandb.finish()




if __name__ == "__main__":

    parser = get_model_parser()

    args = parser.parse_args()

    run_dir = args.dir

    df = read_run_dir(run_dir)

    task = "linear_regression"
    #task = "sparse_linear_regression"
    #task = "decision_tree"
    #task = "relu_2nn_regression"

    run_id = args.runid # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    visualize_ov_from_data(run_path)  # these are normally precomputed at the end of training
