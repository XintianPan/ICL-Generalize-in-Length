from models import TransformerModelStackTogether
import os
import wandb
import matplotlib.pyplot as plt
import torch 
from eval import get_run_metrics, read_run_dir, get_model_from_run
from visualize_qk import extract_qkv_matrices_head_only, qk_circuit_compute, heatmap_draw_qk_ebmeds
import seaborn as sns
from torch import nn

from args_parser import get_model_parser

from visualize_ov import extract_ou


def visualize_result(matrix, title):
    '''
    Visualize softmax(x @ W_E^T @ W_Q @ W_K^T @ W_E @ x^T)
    '''
    matrix = nn.functional.softmax(matrix, dim=-1)
    matrix_nd = matrix.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_nd, cmap='viridis')
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def compute_attention_v(x, beta, W_e, W_v, matrix):
    '''
    Compute softmax(x @ W_E^T @ W_Q @ W_K^T @ W_E @ x^T) @ x[:, -1:, :] @ W_E^T @ W_V
    '''
    y = torch.einsum('nlx,nxy->nly', x, beta)
    z = torch.cat([x, y], dim = 2)
    z.squeeze(0)
    z[:, -1, -1] = 0
    z = z[:, -1:, :]
    matrix = nn.functional.softmax(matrix, dim=-1)
    W_e_t = W_e.transpose(0, 1)
    matrix_val = matrix @ z @ W_e_t @ W_v
    return matrix_val

def visualize_attention_v(x, beta, W_e, W_v, matrix, title):
    '''
    Visualize softmax(x @ W_E^T @ W_Q @ W_K^T @ W_E @ x^T) @ x @ W_E^T @ W_V
    '''
    matrix_val = compute_attention_v(x, beta, W_e, W_v, matrix)
    matrix_nd = matrix_val.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_nd, cmap='viridis')
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()


def compute_attention_ov(W_u, W_o, matrices):
    '''
    Cat(head0, .., headn) into one and
    Compute (head0, ..., headn) @ W_o @ W_u^T
    '''
    new_matrix = torch.cat(matrices, dim=2)
    W_u_T = W_u.transpose(0, 1)
    return new_matrix @ W_o @ W_u_T

def visualize_attention_ov(W_u, W_o, matrices, title):
    '''
    Visualize (head0, ..., headn) @ W_o @ W_u^T
    '''
    new_matrix = compute_attention_ov(W_u, W_o, matrices)
    new_matrix = new_matrix[0, :, :]

    # new_matrix_nd = new_matrix.cpu().numpy()
    
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(new_matrix_nd, cmap='viridis')
    # plt.title(title)
    # wandb.log({title: wandb.Image(plt)})
    # plt.close()
    return new_matrix[-1, -1]



def f(x, beta, qk_circuit):
    '''
    Compute z @ W_e^T @ W_q @ W_k^T @ W_e @ z[:, -1:, :]^T
    '''
    y = torch.einsum('nlx,nxy->nly', x, beta)
    z1 = torch.cat([x, y], dim = 2)
    z1.squeeze(0)
    z2 = z1.clone()
    z1[:, -1, -1] = 0
    z2[:, -1, -1] = 0
    z2 = z2[:, -1:, :]
    z2 = z2.transpose(-2, -1)

    ret = z1 @ qk_circuit @ z2
    return ret
    
def sample_x(L, n_dims):
    x = torch.randn(1, L, n_dims)
    return x

def sample_beta(L, n_dims):
    beta = torch.randn(1, n_dims, 1) 
    G = torch.zeros(n_dims, 1)
    for i in range(1):
        G[int(i * n_dims):int((i + 1) * n_dims), i] = 1
    
    beta = torch.einsum('nxy,xy->nxy', beta, G)
    return beta

def sample_fix_beta(steps=10, L=100, n_dims=0):
    beta = sample_beta(L, n_dims)
    for i in range(steps):
        title = f"QK Analyze: Fixed Beta - {i}"

def sample_only_change_last_x(qkv, qk_circuit, steps=10, L=100, n_dims=0, head_name=""):
    x_pre = sample_x(L - 1, n_dims)
    beta = sample_beta(L, n_dims)
    W_e = qkv['W_e']
    W_v = qkv['W_v']
    for i in range(steps):
        title = f"QK Analyze: Change Last x " + head_name + f" - step{i}"
        x_last = sample_x(1, n_dims)
        x = torch.cat([x_pre, x_last], dim=1)
        ret = f(x, beta, qk_circuit)
        visualize_attention_v(x, beta, W_e, W_v, ret, title)

def sample_only_change_last_x_ov(model, qkv_matrices, steps=10, L=100, n_dims=0):
    W_o, W_u = extract_ou(model)
    x_pre = sample_x(L - 1, n_dims)
    beta = sample_beta(L, n_dims)
    x_y_data = []
    for i in range(steps):
        title = f"Full attention OV Analyze: Change Last x - step{i}"
        x_last = sample_x(1, n_dims)
        x = torch.cat([x_pre, x_last], dim=1)
        matrices = []
        for qkv in qkv_matrices:
            qk_circuit = qk_circuit_compute(qkv)
            W_e = qkv['W_e']
            W_v = qkv['W_v']
            ret = f(x, beta, qk_circuit)
            ret = compute_attention_v(x, beta, W_e, W_v, ret)
            matrices.append(ret)
        last_true = x[0, -1, -1]
        last_val = visualize_attention_ov(W_u, W_o, matrices, title)
        x_y = (last_val.tolist(), last_true.tolist())
        x_y_data.append(x_y)

    mean_table = wandb.Table(data=x_y_data, columns=["val", "true"])
    mean_title = "True - val graph"
    mean_plot = wandb.plot.scatter(mean_table, x='val', y='true', title=mean_title)
    wandb.log({
        mean_title: mean_plot,
    })


def qk_analyze(run_path, step=-1):
    model, conf = get_model_from_run(run_path=run_path, step=step)
    wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project + "-analyze-QK",
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=conf.wandb.name,
            resume=True,
    )
    model.cuda().eval()
    qkv_matrices = extract_qkv_matrices_head_only(model)
    n_dims = conf.model.n_dims

    
    sample_only_change_last_x_ov(model, qkv_matrices, steps=10000, L=200, n_dims=n_dims)
    # for i, qkv in enumerate(qkv_matrices):
    #     qk_circuit = qk_circuit_compute(qkv)
    #     head_name = f"for Head {i}"
    #     heatmap_draw_qk_ebmeds(qkv, f"Head {i}")
    #     sample_only_change_last_x(qkv, qk_circuit, steps=5, L=40, n_dims=n_dims, head_name=head_name)



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
    qk_analyze(run_path)  # these are normally precomputed at the end of training