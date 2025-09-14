import numpy as np
import copy
from avalanche.core import SupervisedPlugin

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



class TemperatureAnnealingPlugin(SupervisedPlugin):
    """
    Plugin for temperature annealing in Gumbel-Softmax during MultiBand VAE training.
    """
    def __init__(self, update_freq: int = 100):
        super().__init__()
        self.update_freq = update_freq
        self.step_count = 0

    def before_training_iteration(self, strategy, **kwargs):
        """Update temperature before each training iteration."""
        if self.step_count % self.update_freq == 0:
            if hasattr(strategy.model, 'update_temperature'):
                strategy.model.update_temperature()
        self.step_count += 1

    def after_training_exp(self, strategy, **kwargs):
        """Reset step count after each experience."""
        self.step_count = 0