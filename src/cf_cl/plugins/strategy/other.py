import numpy as np

from .simple import BetaAnnealingPlugin
from .multiband_strategy import TemperatureAnnealingPlugin


# Beta annealing variations
def create_cyclical_beta_plugin(cycle_length: int = 1000, 
                               min_beta: float = 0.0, 
                               max_beta: float = 1.0):
    """
    Create a cyclical beta annealing plugin (inspired by cyclical annealing papers).
    """
    
    class CyclicalBetaAnnealingPlugin(BetaAnnealingPlugin):
        def __init__(self):
            super().__init__(initial_beta=min_beta, final_beta=max_beta, 
                           anneal_steps=cycle_length, anneal_type='linear')
            self.cycle_length = cycle_length
            self.min_beta = min_beta
            self.max_beta = max_beta
        
        def _compute_beta(self):
            cycle_position = self.current_step % self.cycle_length
            progress = cycle_position / self.cycle_length
            
            # Linear increase within each cycle
            beta = self.min_beta + progress * (self.max_beta - self.min_beta)
            return float(beta)
    
    return CyclicalBetaAnnealingPlugin()


# Advanced temperature annealing
class AdaptiveTemperaturePlugin(TemperatureAnnealingPlugin):
    """
    Temperature plugin that adapts based on training progress.
    """
    
    def __init__(self, initial_temp: float = 1.0, min_temp: float = 0.5):
        super().__init__(update_freq=100)
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.task_losses = []
    
    def after_training_iteration(self, strategy, **kwargs):
        """Track losses to adapt temperature."""
        if hasattr(strategy, 'loss') and strategy.loss is not None:
            self.task_losses.append(float(strategy.loss))
    
    def after_training_exp(self, strategy, **kwargs):
        """Adapt temperature based on task performance."""
        if self.task_losses:
            avg_loss = np.mean(self.task_losses[-100:])  # Last 100 iterations
            
            # Adaptive temperature: increase if loss is high
            if avg_loss > 1.0:
                strategy.model.temperature = min(self.initial_temp, 
                                               strategy.model.temperature * 1.1)
            else:
                strategy.model.temperature = max(self.min_temp,
                                               strategy.model.temperature * 0.99)
        
        self.task_losses = []  # Reset for next task