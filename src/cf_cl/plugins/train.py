import os

from avalanche.evaluation.metrics import (
    disk_usage_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins import EarlyStoppingPlugin
import torch
from torch.utils.tensorboard import SummaryWriter


from .metrics import  (image_quality_metrics, visual_metrics, 
                       latent_plugs)

from .strategy import (VAEStrategy, MultiBandVAEStrategy, 
                       BetaAnnealingPlugin, TemperatureAnnealingPlugin, 
                       ControlledForgettingRehearsalPlugin)



# ----------------------------
# Continual Learning Strategy
# ----------------------------
def cl_strategy(
    model, optimizer, criterion, device, loggers, cfg_strategy, 
    train_plugins=None, type_strategy='simple', *extra_metrics,
):
    base_metrics = [
        # helpers return iterables; no need to splat here, just collect
        # *gpu_usage_metrics(gpu_id=0, minibatch=True, epoch=True, stream=False),
        *loss_metrics(epoch=True, experience=True, stream=True),
        *disk_usage_metrics(stream=True),
        *timing_metrics(epoch_running=False, epoch=True),
    ]
    custom_metrics = [
            *visual_metrics, *latent_plugs, 
        ]
    if model.img_channels == 3: 
        custom_metrics += [
            *image_quality_metrics(experience=True)
        ]
        
    all_metrics = [*extra_metrics, *custom_metrics, *base_metrics]

    evaluator = EvaluationPlugin(
        *all_metrics,
        loggers=loggers,
    )
    
    early_stop = EarlyStoppingPlugin(
        patience=5,
        val_stream_name="train",
        mode="min",
        metric_name="loss",
        verbose=True,
    )

    beta_vae =  BetaAnnealingPlugin(
            initial_beta=0,
            final_beta=1,
            anneal_steps=10,
            anneal_type='linear'
        )
    
    model.eval()
    dummy = torch.rand(1, model.img_channels, model.img_size, model.img_size)  # dummy input for visualization purposes
    
    if train_plugins is None: 
        train_plugins = [early_stop, beta_vae]
    else:
        train_plugins += [early_stop, beta_vae]

    if type_strategy == 'simple':
        return VAEStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=cfg_strategy.train_mb_size,
            train_epochs=cfg_strategy.train_epochs,
            eval_mb_size=cfg_strategy.eval_mb_size,
            evaluator=evaluator,
            plugins=train_plugins,
            device=device,
        )

    elif type_strategy == 'multiband':        
        train_plugins += [
            TemperatureAnnealingPlugin(update_freq=50),
            ControlledForgettingRehearsalPlugin()
        ]
        
        strategy = MultiBandVAEStrategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=evaluator,
            train_mb_size=cfg_strategy.train_mb_size,
            train_epochs=cfg_strategy.train_epochs,
            eval_mb_size=cfg_strategy.eval_mb_size,
            plugins=train_plugins,
            device=device,
        )
        
        return strategy



# ----------------------------
# Train / Eval Loop
# ----------------------------
def train_tasks(benchmark, cl_strategy, exp_name="Naive", **kwargs):
    print(f"Starting experiment {exp_name}...")
    all_results = []

    for exp in benchmark.train_stream:
        print(f"Start of experience: {exp.current_experience}")
        print(f"Current classes: {exp.classes_in_this_experience}")

        # Train on this experience + replay buffer
        cl_strategy.train(
            exp,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=32,
            num_workers=8,
            **kwargs,
        )
        print("Training Experience completed")

        # Evaluate on the whole test stream
        print("Computing metrics on the whole test set...")
        eval_results = cl_strategy.eval(benchmark.test_stream)
        all_results.append(eval_results)

    print("Experiment completed.")

    return all_results
