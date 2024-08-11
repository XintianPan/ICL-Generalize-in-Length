from collections import OrderedDict
import re
import os

import pandas as pd
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run

if __name__ == "__main__":

    run_dir = "/home/xc425/project/models"

    df = read_run_dir(run_dir)

    task = "linear_regression"
    #task = "sparse_linear_regression"
    #task = "decision_tree"
    #task = "relu_2nn_regression"

    run_id = "95968066-fd53-4975-b28c-142848e10054"  # if you train more models, replace with the run_id from the table above

    run_path = os.path.join(run_dir, task, run_id)
    get_run_metrics(run_path)  # these are normally precomputed at the end of training