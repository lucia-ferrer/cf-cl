import os
import pathlib
import signal
import argparse

import torch
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import ReplayPlugin

from cf_cl.data.datasets import benchmark_import, config_factory
from cf_cl.models.build_model import build_model_optim
from cf_cl.plugins.train import cl_strategy, train_tasks
from cf_cl.utils import set_seed

nohup_like = signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
torch.cuda.empty_cache()


# -----------------------------
# Config for Experiment
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="SplitMNIST", help="Name of the benchmark")
parser.add_argument("-ni", "--new_instances", action='store_true', help="Use new classes scenario, or new instances scenario")
parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()
name = args.name
ni_scenario = args.new_instances
seed = args.seed

set_seed(seed)
CONFIG = config_factory(name)

# ----------------------------
# Dataset
# ----------------------------
classic_benchmark = benchmark_import(CONFIG, ni_scenario=ni_scenario)


# ----------------------------
# Model / Optim / Loss
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, optimizer, criterion = build_model_optim(model_cfg = CONFIG.model)

# ----------------------------
# Logs
# ----------------------------
log_folder = pathlib.Path().cwd() / "logs"
os.makedirs(log_folder, exist_ok=True)
exp_name = f"{CONFIG.name.lower()}_replay"
if ni_scenario:
    exp_name += "_ni"

loggers = [
    TensorboardLogger(tb_log_dir=log_folder / "tb_data" / exp_name),
]
if not nohup_like:
    loggers.append(InteractiveLogger())

# ----------------------------
# Replay Plugin (Avalanche 0.6.0)
# ----------------------------
# memory_size: total number of patterns kept in the buffer.
replay_plugin = ReplayPlugin(
    mem_size=5000
    # batch_size or batch_size_mem params may exist depending on minor versions; default mixing works well.
)
train_plugins = [replay_plugin]


# ----------------------------
# Continual Learning Strategy
# ----------------------------
train_strategy = cl_strategy(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    loggers=loggers,
    train_plugins=train_plugins,
    cfg_strategy=CONFIG.train
)

# ----------------------------
# Train / Eval Loop
# ----------------------------
train_tasks(classic_benchmark, train_strategy, exp_name=exp_name)
