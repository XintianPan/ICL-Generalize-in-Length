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

import math


def visualize_result(matrix, title):
    '''
    Visualize softmax(x @ W_E^T @ W_Q @ W_K^T @ W_E @ x^T)
    '''
    matrix = nn.functional.softmax(matrix, dim=-1)
    matrix_nd = matrix.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_nd, cmap='viridis', annot=True)
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



def f(x, beta, qk_circuit, embd_dim=16):
    '''
    Compute z @ W_e^T @ W_q @ W_k^T @ W_e @ z[:, -1:, :]^T
    '''
    y = torch.einsum('nlx,nxy->nly', x, beta)
    z1 = torch.cat([x, y], dim = 2)
    z1.squeeze(0)
    z2 = z1.clone()
    z1[:, -1, -1] = 0
    z2[:, -1, -1] = 0
    # z1 = z1[:, -1, :]
    z2 = z2.transpose(-2, -1)

    ret = z1 @ qk_circuit @ z2
    ret = ret / math.sqrt(embd_dim)
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

def softmax_score_cal(qkv, L=20, n_dims=10, embd_dim=16):
    x = sample_x(L, n_dims=n_dims)
    beta = sample_beta(L, n_dims=n_dims)
    qk_circuit = qk_circuit_compute(qkv)
    rel = f(x, beta, qk_circuit, embd_dim=embd_dim)
    return rel

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
    n_embd = conf.model.n_embd // 4

    
    for i, qkv in enumerate(qkv_matrices):
        print(i)
        head_name = f"Softmax Score for Head {i}"
        heatmap_draw_qk_ebmeds(qkv, f"QK circuit for Head {i}")
        rel = softmax_score_cal(qkv, L=20, n_dims=n_dims, embd_dim=n_embd)
        visualize_result(rel[0, :, :], head_name)




if __name__ == "__main__":

    parser = get_model_parser()

    args = parser.parse_args()

    run_dir = args.dir

    task = "linear_regression"
    #task = "sparse_linear_regression"
    #task = "decision_tree"
    #task = "relu_2nn_regression"

    run_id = args.runid # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    qk_analyze(run_path)  # these are normally precomputed at the end of training