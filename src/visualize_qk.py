from models import TransformerModelStackTogether
import os
import wandb
import matplotlib.pyplot as plt
import torch 
from eval import get_run_metrics, read_run_dir, get_model_from_run
import seaborn as sns

from args_parser import get_model_parser

def extract_qkv_matrices(model):
    # Extract the embedding matrix
    
    W_e = model._read_in.weight.detach().cpu() # shape (hidden_size, input_dim) input_dim = n_dims + 1
    
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

    qkv_matrices = []


    n_dim = W_e.shape[1]

    # Reshape to separate heads
    W_q = W_q.view(hidden_size, num_heads, head_dim)
    W_k = W_k.view(hidden_size, num_heads, head_dim)
    W_v = W_v.view(hidden_size, num_heads, head_dim)


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


        qkv_matrices.append({
            'W_q': W_q_head,
            'W_k': W_k_head,
            'W_v': W_v_head,
            'b_q': b_q_head,
            'b_k': b_k_head,
            'b_v': b_v_head,
            'W_e': W_e
        })

    return qkv_matrices

def extract_qkv_matrices_head_only(model):
    # Extract the embedding matrix
    
    W_e = model._read_in.weight.detach().cpu() # shape (hidden_size, input_dim) input_dim = n_dims + 1
    
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

    qkv_matrices = []

    n_dim = W_e.shape[1]

    # Reshape to separate heads
    W_q = W_q.view(hidden_size, num_heads, head_dim)
    W_k = W_k.view(hidden_size, num_heads, head_dim)
    W_v = W_v.view(hidden_size, num_heads, head_dim)


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


        qkv_matrices.append({
            'W_q': W_q_head,
            'W_k': W_k_head,
            'W_v': W_v_head,
            'b_q': b_q_head,
            'b_k': b_k_head,
            'b_v': b_v_head,
            'W_e': W_e
        })

    return qkv_matrices

def qk_circuit_compute(qkv_matrices):
    W_q = qkv_matrices['W_q']
    W_k = qkv_matrices['W_k']
    W_e = qkv_matrices['W_e']
    W_e_t = W_e.transpose(0, 1)
    W_k_t = W_k.transpose(0, 1)

    return W_e_t @ W_q @ W_k_t @ W_e

def heatmap_draw_qk(qkv_matrices, title):
    Wq = qkv_matrices['W_q']
    Wk = qkv_matrices['W_k']
    Wk_t = Wk.transpose(0, 1)
    QK_matrix = torch.matmul(Wq, Wk_t)
    my_array = QK_matrix.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(my_array, cmap='viridis', annot=True)
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def heatmap_draw_qk_ebmeds(qkv_matrices, title):
    # We want to visualize QK ciruit
    # Then output matrix is W_e^T * W_q * W_k^T * W_e
    Wq = qkv_matrices['W_q']
    Wk = qkv_matrices['W_k']
    We = qkv_matrices['W_e']

    _, norm_dim = Wq.shape

    Wk_t = Wk.transpose(0, 1)
    We_t = We.transpose(0, 1)
    QK_circuit = torch.matmul(We_t, Wq)
    QK_circuit = torch.matmul(QK_circuit, Wk_t)
    QK_circuit = torch.matmul(QK_circuit, We)
    QK_circuit = QK_circuit * ((norm_dim) ** -5)

    my_array = QK_circuit.cpu().numpy()

    plt.figure(figsize=(10 * 2, 8 * 2))
    sns.heatmap(my_array, cmap='viridis', annot=True)
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def visualize_qk_from_data(run_path, step=-1):
    model, conf = get_model_from_run(run_path, step)
    wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project + "-vis-QK",
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=conf.wandb.name,
            resume=True,
    )
    model = model.cuda().eval()

    qkv_matrices = extract_qkv_matrices(model)
    embeds = model._read_in.weight.detach().cpu()
    for i in range(conf.model.n_head):
        title = f"QK matrix heatmap for head {i}"
        
        # heatmap_draw_qk(qkv_matrices[i], title)
        heatmap_draw_qk_ebmeds(qkv_matrices[i], title)

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
    visualize_qk_from_data(run_path)  # these are normally precomputed at the end of training
