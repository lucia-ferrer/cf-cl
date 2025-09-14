from typing import TypeVar
import numpy as np
from torch.utils.data import Subset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.scenarios import CLScenario
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks import with_classes_timeline

def dirichlet_task_labels(labels, num_experiences, alpha=0.5):
    """
    Assigns each sample a task label (experience index) using Dirichlet sampling.
    
    Args:
        labels: Array of sample class labels.
        num_experiences: Number of experiences.
        alpha: Dirichlet concentration parameter.
        
    Returns:
        task_labels: Array where task_labels[i] is the experience for sample i.
    """
    task_labels:list[int|None] = [None] * len(labels)
    classes = np.unique(labels)
    for cls in classes:
        indices = np.where(labels == cls)[0]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_experiences)
        counts = np.round(proportions * len(indices)).astype(int)
        diff = len(indices) - np.sum(counts)
        counts[:diff] += 1
        start = 0
        for exp_id, c in enumerate(counts):
            for i in indices[start:start + c]:
                task_labels[i] = exp_id
            start += c
    return task_labels

def create_dirichlet_ni_scenario(*, 
                                 dataset:AvalancheDataset, 
                                 data_dir:str, 
                                 num_experiences:int=5, 
                                 alpha=0.5,
                                 train_transform=None,
                                 eval_transform=None,
                                 **kwargs)->CLScenario:
    """
    Create a New Instances scenario with Dirichlet-partitioned experiences.
    
    Args:
        dataset: Class module of the dataset ('cifar10', 'mnist', etc.)
        num_experiences: Number of experiences in the scenario
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility
    
    Returns:
        Avalanche benchmark with Dirichlet-partitioned experiences
    """    
    # Load dataset
    train_dataset = dataset(data_dir, train=True, transform=train_transform)
    test_dataset = dataset(data_dir, train=False, transform=eval_transform)
        
    # Extract labels for partitioning
    train_labels = train_dataset.targets
    
    # Create Dirichlet partitions
    task_labels = dirichlet_task_labels(
        labels=train_labels,
        num_experiences=num_experiences,
        alpha=alpha
    )
    
    experience_datasets = []
    for exp_id in range(num_experiences):
        indices = [i for i, t in enumerate(task_labels) if t == exp_id]
        subset = Subset(train_dataset, indices)
        exp_dataset = _make_taskaware_classification_dataset(
            subset.dataset,
            # task_labels = [exp_id]*len(subset.dataset)  # TODO: verify if i need this
        )
        experience_datasets.append(exp_dataset)

    benchmark = benchmark_from_datasets(
        train=experience_datasets,
        test=[_make_taskaware_classification_dataset(test_dataset)]
    )
    
    if kwargs.get('return_task_id', True):
        return with_classes_timeline(benchmark)
    else:
        return benchmark
