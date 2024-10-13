import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics, baseline_names
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from dataset_base import DatasetBase
from torch.utils.data import DataLoader
from data_linear import LinearReg
from tasks import squared_error, mean_squared_error
from visualize_both import visualize_both
from cpl import CPL

import wandb

from weight_tying import tie_head_weights_qk

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, count_id=None, loss_only_zero=False, weight_tying=False):
    if not loss_only_zero:
        if count_id is None:
            if weight_tying:
                tie_head_weights_qk(model)
            optimizer.zero_grad()
            output = model(xs, ys)
            loss = loss_func(output, ys)
            loss.backward()
            optimizer.step()
        else:
            xs[:, count_id, -1] = 0
            optimizer.zero_grad()
            output = model(xs, ys)
            loss = loss_func(output, ys)
            loss.backward()
            optimizer.step()
    else:
        assert count_id is not None
        if weight_tying:
            tie_head_weights_qk(model)
        xs[:, count_id, -1] = 0
        optimizer.zero_grad()
        output = model(xs, ys)
        loss = loss_func(output[:, count_id], ys[:, count_id])
        loss.backward()
        optimizer.step()

    return loss.detach().item(), output.detach()

def validate_step(model, xs, ys, loss_func, count_id=None, loss_only_zero=False):
    if not loss_only_zero:
        if count_id is None:
            output = model(xs, ys)
            loss = loss_func(output, ys)
        else:
            xs[:, count_id, -1] = 0
            output = model(xs, ys)
            loss = loss_func(output, ys)
    else:
        assert count_id is not None
        xs[:, count_id, -1] = 0
        output = model(xs, ys)
        loss = loss_func(output[:, count_id], ys[:, count_id])        
    return loss.detach().item()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    # Add Slurm checkpoint handler

    cpl = CPL()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    # title_name = "testingnorm" + str(args.model.n_dims) + "d/index_to_file_dict.yaml"

    # data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), title_name)

    L = 10 * n_dims

    # fp = open(data_dir)

    # index_to_file_dict = yaml.load(fp)

    train_method = LinearReg({"L": L, "dx": n_dims, "dy": 1, "number_of_samples": 64, "noise_std": 0.01})

    val_method = LinearReg({"L": L, "dx": n_dims, "dy": 1, "number_of_samples": 16, "noise_std": 0.01})

    # training_dataset = DatasetBase(
    #     index_to_file_dict=index_to_file_dict["train"], 
    #     data_method=data_method,
    #     data_method_args_dict={"L": 8 * n_dims},
    #     load_data_into_memory=True
    # )

    # validating_dataset = DatasetBase(
    #     index_to_file_dict=index_to_file_dict["val"], 
    #     data_method=data_method,
    #     data_method_args_dict={"L": 8 * n_dims},
    #     load_data_into_memory=True
    # )

    # train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=False)

    # train_iterator = iter(train_dataloader)

    # val_dataloader = DataLoader(validating_dataset, batch_size=16, shuffle=True)

    # val_iterator = iter(val_dataloader)

    num_training_examples = args.training.num_training_examples

    tying_flag = args.model.weight_tying

    # counter = L - 1
    # xs, ys, xv, yv = None, None, None, None
    # arr = [i for i in range(0, L)]

    loss_sec = args.training.loss_only_zero

    for i in pbar:
        # data_sampler_args = {}
        # task_sampler_args = {}

        # if "sparse" in args.training.task:
        #     task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        # if num_training_examples is not None:
        #     assert num_training_examples >= bsize
        #     seeds = sample_seeds(num_training_examples, bsize)
        #     data_sampler_args["seeds"] = seeds
        #     task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # xs = data_sampler.sample_xs(
        #     curriculum.n_points,
        #     bsize,
        #     curriculum.n_dims_truncated,
        #     **data_sampler_args,
        # )
        # task = task_sampler(**task_sampler_args)
        # ys = task.evaluate(xs)

        # counter += 1
        # if counter == L:
        #     xs, ys = next(train_iterator)
        #     xv, yv = next(val_iterator)
        #     counter = 0
        #     import random
        #     random.shuffle(arr)

        train_data = train_method.__generatedata__()

        xs, ys = train_method.__transform__(train_data)

        val_data = val_method.__generatedata__()

        xv, yv = val_method.__transform__(val_data)

        loss_func = mean_squared_error


        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, None, loss_sec, weight_tying=tying_flag)

        val_loss = validate_step(model, xv.cuda(), yv.cuda(), loss_func, None, loss_sec)

        point_wise_tags = list(range(L))
        point_wise_loss_func = squared_error
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_train_loss": loss,
                    "validation_loss": val_loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if cpl.check():
            print()
            print('='*100)
            print('detected preemption flag inside training loop')
            print('exiting gracefully (saving model checkpoint, etc.) ...')
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))
            wandb.finish()
            print('exiting now')
            print('='*100)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    wandb.finish()

    if not args.test_run:
        visualize_both(args.out_dir)
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval



if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
