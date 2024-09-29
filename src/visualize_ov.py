from models import TransformerModelStackTogether
import os
import wandb
import matplotlib.pyplot as plt
import torch 
from eval import get_run_metrics, read_run_dir, get_model_from_run
import seaborn as sns

from args_parser import get_model_parser

def extract_ov_circuits(model):
    # Step 1: Extract W_E
    W_E = model._read_in.weight  # Shape: [n_embd, n_input_dim]
    b_E = model._read_in.bias    # Shape: [n_embd]

    # Transpose W_E
    W_E_T = W_E.T  # Shape: [n_input_dim, n_embd]

    # Step 2: Access the transformer block and attention module
    transformer_block = model._backbone.h[0]
    attn = transformer_block.attn

    # Step 3: Extract W_V
    c_attn_weight = attn.c_attn.weight  # Shape: [3 * n_embd, n_embd]
    c_attn_bias = attn.c_attn.bias      # Shape: [3 * n_embd]

    n_embd = model._backbone.config.n_embd
    W_V = c_attn_weight[:, 2 * n_embd:].T  # Shape: [n_embd, n_embd]
    b_V = c_attn_bias[2 * n_embd :]         # Shape: [n_embd]

    # Step 4: Extract W_O
    W_O = attn.c_proj.weight.T  # Shape: [n_embd, n_embd]
    b_O = attn.c_proj.bias      # Shape: [n_embd]

    # Step 5: Extract W_R
    W_R = model._read_out.weight.T  # Shape: [n_embd, 1]
    b_R = model._read_out.bias      # Shape: [1]

    # Step 6: Store total matrix
    ov_circuits = []
    ov_circuits.append(
        {

            'W_E_T': W_E_T,
            'W_O': W_O.clone(),
            'W_V': W_V.clone(),
            'W_R': W_R,
        }
    )

    # Step 7: Define head dimensions
    n_head = model._backbone.config.n_head
    head_dim = n_embd // n_head

    # Step 8: Reshape W_V and W_O to separate heads
    W_V_heads = W_V.view(n_embd, n_head, head_dim)  # Shape: [n_embd, n_head, head_dim]
    W_O_heads = W_O.view(n_head, head_dim, n_embd)
    
    # Step 9: Store Each Head's OV circuit
    for head in range(n_head):
        W_V_head = W_V_heads[:, head, :]  # [n_embd, head_dim]
        W_O_head = W_O_heads[head, :, :]  # [n_embd, head_dim]
        ov_circuits.append(
            {

                'W_E_T': W_E_T,
                'W_O': W_O_head,
                'W_V': W_V_head,
                'W_R': W_R,
            }
        )


    return ov_circuits

def heatmap_draw_ov(ov_circuits, title):
    # We want to visualize OV ciruit
    # Then output circuit is W_E_T @ W_V @ W_O @ W_R
    W_E_T = ov_circuits['W_E_T']
    W_V = ov_circuits['W_V']
    W_O = ov_circuits['W_O']
    W_R = ov_circuits['W_R']

    OV_circuit = W_E_T @ W_V @ W_O @ W_R

    my_array = OV_circuit.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(my_array, cmap='viridis')
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def visualize_ov_from_data(run_path, step=-1):
    model, conf = get_model_from_run(run_path, step)
    wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project + "-vis-OV",
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=conf.wandb.name,
            resume=True,
    )
    model = model.cuda().eval()

    ov_circuits = extract_ov_circuits(model)
    for i in range(conf.model.n_head + 1):
        title = None
        if i == 0:
            title = "OV circuit heatmap"
        else:
            title = f"OV circuit heatmap for head {i - 1}"
        
        heatmap_draw_ov(ov_circuits[i], title)

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