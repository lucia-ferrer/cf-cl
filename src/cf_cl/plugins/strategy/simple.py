from typing import Optional
from avalanche.training.supervised import VAETraining
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin

import torch
from torch.utils.data import DataLoader

from vae_ls_align.models.vae import vae_loss

# ---------------------
# Dataloader special
# ---------------------
class VAEStrategy(VAETraining):

    def __init__(self, *,
                 model,
                 optimizer,
                 criterion = vae_loss,
                 train_mb_size: int = 32,
                 train_epochs: int = 10,
                 eval_mb_size: Optional[int] = None,
                 device='cpu',
                 plugins: Optional[list[SupervisedPlugin]] = None,
                 evaluator=EvaluationPlugin(),
                 eval_every: int = -1,
                 **kwargs):
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **kwargs
        )
        
        # Initialize beta for annealing
        self.beta = 1.0
    
    def criterion(self):
        losses = self._criterion(self.mb_x, self.mb_output, beta=self.beta)
        return losses
    
    def make_train_dataloader(self, 
                               num_workers=16,
                                shuffle=True,
                                pin_memory=True,
                                persistent_workers:Optional[bool]=True,
                                drop_last=False,
                                **kwargs):
        self.dataloader = DataLoader(
            self.adapted_dataset,
            shuffle=shuffle,
            batch_size=self.train_mb_size,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if persistent_workers is not None else False,
            prefetch_factor=kwargs.get("prefetch_factor", 8),
            num_workers=num_workers,
            # os.cpu_count() // 4 if torch.cuda.is_available() else 0,
        )


class BetaAnnealingPlugin(SupervisedPlugin):
    """
    Plugin for beta annealing in VAE loss (KL divergence weighting).
    """
    
    def __init__(self, 
                 initial_beta: float = 0.0,
                 final_beta: float = 1.0,
                 anneal_steps: int = 5000,
                 anneal_type: str = 'linear'):
        super().__init__()
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.anneal_steps = anneal_steps
        self.anneal_type = anneal_type
        self.current_step = 0
        self.current_beta = initial_beta

    def _compute_beta(self):
        """Compute current beta value based on annealing schedule."""
        if self.current_step >= self.anneal_steps:
            return self.final_beta
        
        progress = self.current_step / self.anneal_steps
        
        if self.anneal_type == 'linear':
            beta = self.initial_beta + progress * (self.final_beta - self.initial_beta)
        elif self.anneal_type == 'cosine':
            beta = self.initial_beta + 0.5 * (self.final_beta - self.initial_beta) * \
                   (1 - torch.cos(torch.tensor(progress * torch.pi)))
        elif self.anneal_type == 'exponential':
            beta = self.initial_beta * (self.final_beta / self.initial_beta) ** progress
        else:
            raise ValueError(f"Unknown anneal_type: {self.anneal_type}")
        
        return float(beta)

    def before_training_iteration(self, strategy, **kwargs):
        """Update beta before each training iteration."""
        self.current_beta = self._compute_beta()
        
        strategy.beta = self.current_beta
        self.current_step += 1

    def after_training_exp(self, strategy, **kwargs):
        """Reset step count after each experience (optional)."""
        self.current_step = 0
