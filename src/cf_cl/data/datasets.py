import importlib
import inspect
import sys

import torch
from avalanche.benchmarks import CLScenario, NCScenario, nc_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from torchvision import transforms
from torchvision.transforms import Compose

from .config import (
    Config,
    ConfigBenchmark, 
    ConfigModel,
    ConfigStrategy
)
from .newinstances_datasets import create_dirichlet_ni_scenario


#-------------------------
# DATASETS CONFIG 
#------------------------
_empty_transform = Compose([]) 
_to_tensor_transform = Compose([transforms.ToTensor(),]) # Keeps data bounded in between [0,1]
_standarize_transform = Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
_aug_train_transform = Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )

_aug_train_resize_transform = Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )
_default_cifar10_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], 
        std=[0.2470, 0.2435, 0.2616]
    )
    ]
)



CLASSIC_BENCHMARKS = {
    "SplitMNIST": ConfigBenchmark(
        n_experiences=5,
        return_task_id=True,
        train_transform=_empty_transform,
        eval_transform=_empty_transform,
    ),
    "SplitOmniglot": ConfigBenchmark(
        n_experiences=4,
        train_transform=_empty_transform,
        eval_transform=_empty_transform,
    ),
    "SplitCIFAR10": ConfigBenchmark(
        n_experiences=5,
        return_task_id=True,
        train_transform=_aug_train_transform,
        eval_transform=_to_tensor_transform,
    ),
    "SplitTinyImageNet": ConfigBenchmark(
        n_experiences=5,
        return_task_id=True,
        train_transform=_aug_train_resize_transform,
        eval_transform=_to_tensor_transform,
    ),
    
}

#------------------
# MODEL PARAMS 
#-------------------
CLASSIC_MODEL = {
    "SplitMNIST": ConfigModel(
        img_channels=1,
        img_size=28,
        latent_dim=2,
        activation="sigmoid",
        categorical_dim=4,
    ),
    "SplitOmniglot": ConfigModel(
        img_channels=1,
        img_size=32,
        latent_dim=32,
    ),
    "SplitCIFAR10": ConfigModel(
        img_channels=3,
        img_size=32,
        latent_dim=128,
        activation="sigmoid",
        base_ch=64,        
    ),
    "SplitTinyImageNet": ConfigModel(
        img_channels=3,
        img_size=256,
        latent_dim=64,
        base_ch = 64,
    ),
}


#------------------
# TRAIN PARAMS 
#------------------

basic_strategy = ConfigStrategy(train_mb_size=32, eval_mb_size=512, train_epochs=10)
medium_strategy = ConfigStrategy(train_mb_size=64, eval_mb_size=512, train_epochs=100)
complex_strategy = ConfigStrategy(train_mb_size=128, eval_mb_size=512, train_epochs=200)

CLASSIC_STRATEGIES = {
    "SplitMNIST": basic_strategy, 
    "SplitCIFAR10": medium_strategy, 
    "SplitOmniglot": medium_strategy,
    "SplitTinyImageNet": complex_strategy, 
}


#-------------------
#  AUX METHODS 
#--------------------
def config_factory(name: str) -> Config:
    """Factory function to create a Config object based on the provided name

    Args:
        name (str): Name of the benchmark

    Returns:
        Config: Config object with benchmark and model configurations
    """
    return Config(
        name=name, 
        benchmark=CLASSIC_BENCHMARKS[name],
        model=CLASSIC_MODEL[name],
        train=CLASSIC_STRATEGIES[name]
    )


def benchmark_import(cfg: Config, ni_scenario: bool = False, seed=None) -> CLScenario:
    """Builds and imports the Classic Benchmark with the arguments from the config

    Args:
        cfg (Config): Config with attr benchmark with configuration
        ni_scenario (bool, optional): Flag to create New Instances scenario. Defaults to False.

    Returns:
        CLScenario: New Classes or New Instances Scenario from Avalanche
    """
    if not ni_scenario :
        mod = importlib.import_module("avalanche.benchmarks.classic")
   
    else:
        cfg.name = cfg.name.replace("Split", "")
        mod = importlib.import_module("avalanche.benchmarks.datasets")
    
    data_dir = default_dataset_location(cfg.name.lower())
    benchmark = getattr(mod, cfg.name)
    args = cfg.benchmark.to_kwargs()
    n_experiences = int(cfg.benchmark.n_experiences)
    
    if not ni_scenario:
        return benchmark(n_experiences, **args)
    
    else:
        if args['train_transform'].transforms == []:
            args['train_transform'] = _to_tensor_transform
            args['eval_transform'] = _to_tensor_transform
                    
        return create_dirichlet_ni_scenario(dataset=benchmark,
                                            data_dir=data_dir,
                                            num_experiences=n_experiences,
                                            **args
                                            )
        
