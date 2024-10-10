from collections import OrderedDict
import re
import os

import pandas as pd
import torch
from tqdm.notebook import tqdm

from args_parser import get_model_parser

from eval import get_run_metrics, read_run_dir, get_model_from_run

if __name__ == "__main__":
    parser = get_model_parser()

    args = parser.parse_args()

    run_dir = args.dir

    df = read_run_dir(run_dir)

    step = args.step

    task = "linear_regression"

    run_id = args.runid  # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    get_run_metrics(run_path, step=step)  # these are normally precomputed at the end of training