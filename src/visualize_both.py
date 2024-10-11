from args_parser import get_model_parser
import os
from eval import get_model_from_run, read_run_dir
from visualize_ov import extract_ov_matrices, heatmap_draw_ov_ebmeds
from visualize_qk import extract_qkv_matrices, heatmap_draw_qk_ebmeds
import wandb


def visualize_both(run_path, step=-1):
    model, conf = get_model_from_run(run_path, step)
    wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project + "-vis-All",
            entity=conf.wandb.entity,
            config=conf.__dict__,
            notes=conf.wandb.notes,
            name=conf.wandb.name,
            resume=True,
    )
    model = model.cuda().eval()

    ov_matrices = extract_ov_matrices(model)
    qkv_matrices = extract_qkv_matrices(model)

    for i, (ov, qk) in enumerate(zip(ov_matrices, qkv_matrices)):
        title_ov = f"OV circuit for head {i}"
        title_qk = f"QK circuit for head {i}"

        heatmap_draw_ov_ebmeds(ov, title_ov)
        heatmap_draw_qk_ebmeds(qk, title_qk)
    # title = "OV circuit"
    
    # title = "OV matrix"
    # heatmap_draw_ov(ov_matrices, title)

    wandb.finish()


if __name__ == "__main__":

    parser = get_model_parser()

    args = parser.parse_args()

    run_dir = args.dir

    df = read_run_dir(run_dir)

    step = args.step

    task = "linear_regression"
    #task = "sparse_linear_regression"
    #task = "decision_tree"
    #task = "relu_2nn_regression"

    run_id = args.runid # if you train more models, replace with the run_id from the table above

    print(step)

    run_path = os.path.join(run_dir, task, run_id)
    visualize_both(run_path, step=step)  # these are normally precomputed at the end of training
