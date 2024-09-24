from models import TransformerModelStackTogether
import os
import wandb
import matplotlib.pyplot as plt
import torch 
from eval import get_run_metrics, read_run_dir, get_model_from_run
import seaborn as sns

def extract_qkv_matrices(model):
    # Access the only transformer block
    W_e = model._read_in.weight.detach().cpu()
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
    qkv_matrices.append(
        {
            'W_q': W_q.clone(),
            'W_k': W_k.clone(),
            'W_v': W_v.clone(),
            'b_q': b_q.clone(),
            'b_k': b_k.clone(),
            'b_v': b_v.clone(),
            'W_e': W_e.clone(),
        }
    )

    n_dim = W_e.shape[1]

    # Reshape to separate heads
    W_q = W_q.view(hidden_size, num_heads, head_dim)
    W_k = W_k.view(hidden_size, num_heads, head_dim)
    W_v = W_v.view(hidden_size, num_heads, head_dim)
    W_e = W_e.view(head_dim, num_heads, n_dim)

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

        W_e_head = W_e[:, head_idx, :]

        qkv_matrices.append({
            'W_q': W_q_head,
            'W_k': W_k_head,
            'W_v': W_v_head,
            'b_q': b_q_head,
            'b_k': b_k_head,
            'b_v': b_v_head,
            'W_e': W_e_head
        })

    return qkv_matrices

def heatmap_draw_qk(qkv_matrices, title):
    Wq = qkv_matrices['W_q']
    Wk = qkv_matrices['W_k']
    Wq_t = Wq.transpose(0, 1)
    QK_matrix = torch.matmul(Wq_t, Wk)
    # dim0, dim1 = QK_matrix.shape
    my_array = QK_matrix.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(my_array, cmap='viridis')
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def heatmap_draw_qk_ebmeds(qkv_matrices, title):
    Wq = qkv_matrices['W_q']
    Wk = qkv_matrices['W_k']
    We = qkv_matrices['W_e']
    Wq_t = Wq.transpose(0, 1)
    We_t = We.transpose(0, 1)
    QK_circuit = torch.matmul(We_t, Wq_t)
    QK_circuit = torch.matmul(QK_circuit, Wk)
    QK_circuit = torch.matmul(QK_circuit, We)
    # dim0, dim1 = QK_matrix.shape
    my_array = QK_circuit.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(my_array, cmap='viridis')
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def visualize_from_data(run_path, step=-1):
    model, conf = get_model_from_run(run_path, step)
    wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project + "-test",
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=conf.wandb.name,
            resume=True,
    )
    model = model.cuda().eval()

    qkv_matrices = extract_qkv_matrices(model)
    embeds = model._read_in.weight.detach().cpu()
    for i in range(conf.model.n_head + 1):
        title = None
        if i == 0:
            title = "QK matrix heatmap"
        else:
            title = f"QK matrix heatmap for head {i - 1}"
        
        heatmap_draw_qk(qkv_matrices[i], title)
        heatmap_draw_qk_ebmeds(qkv_matrices[i], title + " mult embeds")

    wandb.finish()




if __name__ == "__main__":

    run_dir = "/home/xc425/project/models"

    df = read_run_dir(run_dir)

    task = "linear_regression"
    #task = "sparse_linear_regression"
    #task = "decision_tree"
    #task = "relu_2nn_regression"

    run_id = "stackxy_model_one-four_noise15d_nolyaernormandattnnorm_2"  # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    visualize_from_data(run_path)  # these are normally precomputed at the end of training
