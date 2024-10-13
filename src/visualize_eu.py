from models import TransformerModelStackTogether
import os
import wandb
import matplotlib.pyplot as plt
import torch 
from eval import get_run_metrics, read_run_dir, get_model_from_run
import seaborn as sns

from args_parser import get_model_parser



def heatmap_draw_eu(model, title):
    We = model._read_in.weight.detach().cpu().transpose(0, 1)
    Wu = model._read_out.weight.detach().cpu().transpose(0, 1)
    EU_matrix = torch.matmul(We, Wu)
    my_array = EU_matrix.cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(my_array, cmap='viridis', annot=True)
    plt.title(title)
    wandb.log({title: wandb.Image(plt)})
    plt.close()


def visualize_qk_from_data(run_path, step=-1):
    model, conf = get_model_from_run(run_path, step)
    wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project + "-vis-EU",
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=conf.wandb.name,
            resume=True,
    )
    model = model.cuda().eval()

    heatmap_draw_eu(model, "EU")

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
