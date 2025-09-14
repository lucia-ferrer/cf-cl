# MultiBand VAE Strategy and Plugins for Avalanche 0.6.0
# Following the exact two-phase algorithm from the IJCAI 2022 paper

from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin



class ControlledForgettingRehearsalPlugin(SupervisedPlugin):
    """
    Plugin implementing the COMPLETE controlled forgetting mechanism from MultiBand VAE paper.
    This includes similarity-based substitution of replay targets.
    """
    def __init__(self,
                 rehearsal_samples: int = 100,
                 rehearsal_batch_size: int = 32,
                 similarity_threshold: float = 0.9):
        super().__init__()
        self.rehearsal_samples = rehearsal_samples
        self.rehearsal_batch_size = rehearsal_batch_size
        self.similarity_threshold = similarity_threshold  # gamma from paper
        self.replay_data = []

    def after_training_exp(self, strategy, **kwargs):
        """Generate and store replay samples after each experience."""
        if hasattr(strategy.model, 'frozen_translators') and strategy.model.frozen_translators:
            # Generate replay samples for all previous tasks
            for task_id in range(len(strategy.model.frozen_translators)):
                replay_samples, lambda_samples = strategy.model.generate_replay_samples(
                    self.rehearsal_samples, task_id
                )
                if replay_samples is not None:
                    # Store replay data for each previous task
                    self.replay_data.append({
                        'task_id': task_id,
                        'replay_samples': replay_samples,
                        'lambda_samples': lambda_samples
                    })

    def before_training_exp(self, strategy, **kwargs):
        """
        PHASE 2: Global Consolidation with Controlled Forgetting
        Mix replay data with current data, applying controlled forgetting mechanism.
        """
        if self.replay_data and hasattr(strategy, 'adapted_dataset'):
            current_task_id = strategy.model.current_task - 1
            
            # Get current task data representations in global latent space
            current_data_loader = DataLoader(strategy.adapted_dataset, 
                                           batch_size=self.rehearsal_batch_size, 
                                           shuffle=False)
            
            current_global_representations = []
            current_original_data = []
            
            # Collect current task data and their global representations
            strategy.model.eval()
            with torch.no_grad():
                for batch_data, _, task_labels in current_data_loader:
                    batch_data = batch_data.to(strategy.device)
                    # Get global latent representations for current data
                    z_global = strategy.model.get_global_latent_representations(
                        batch_data, current_task_id
                    )
                    current_global_representations.append(z_global)
                    current_original_data.append(batch_data)
            
            if current_global_representations:
                current_global_z = torch.cat(current_global_representations, dim=0)
                current_original_x = torch.cat(current_original_data, dim=0)
                
                # Apply controlled forgetting to replay samples
                updated_replay_data = []
                
                for replay_batch in self.replay_data:
                    task_id = replay_batch['task_id']
                    replay_samples = replay_batch['replay_samples']
                    lambda_samples = replay_batch['lambda_samples']
                    
                    # Get global representations for replay samples
                    replay_global_z = strategy.model.frozen_translators[task_id](
                        lambda_samples, None, task_id
                    )
                    
                    # Apply controlled forgetting: similarity-based substitution
                    substituted_targets = []
                    for i, replay_z in enumerate(replay_global_z):
                        # Calculate cosine similarity with all current data
                        similarities = F.cosine_similarity(
                            replay_z.unsqueeze(0), 
                            current_global_z, 
                            dim=1
                        )
                        max_similarity = torch.max(similarities)
                        
                        if max_similarity >= self.similarity_threshold:
                            # Substitute with most similar current data
                            best_match_idx = torch.argmax(similarities)
                            substituted_targets.append(current_original_x[best_match_idx])
                        else:
                            # Keep original replay sample
                            substituted_targets.append(replay_samples[i])
                    
                    updated_replay_data.extend(substituted_targets)
                
                # Create augmented dataset with controlled forgetting applied
                if updated_replay_data:
                    replay_tensor = torch.stack(updated_replay_data)
                    replay_labels = torch.zeros(len(updated_replay_data))  # Dummy labels
                    replay_task_labels = torch.zeros(len(updated_replay_data))  # Dummy task labels
                    
                    replay_dataset = TensorDataset(replay_tensor, replay_labels, replay_task_labels)
                    
                    # Combine with current dataset
                    strategy.adapted_dataset = ConcatDataset([
                        strategy.adapted_dataset, 
                        replay_dataset
                    ])
            
            strategy.model.train()

    def _calculate_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> float:
        """Calculate cosine similarity between two latent representations."""
        return F.cosine_similarity(z1, z2, dim=0).item()


class MultiBandVAEStrategy(SupervisedTemplate):
    """
    MultiBand VAE training strategy implementing the two-phase algorithm:
    Phase 1: Local training (handled by model adaptation)
    Phase 2: Global consolidation (handled by plugins and strategy)
    """
    def __init__(self,
                 *,
                 model,
                 optimizer,
                 criterion,
                 evaluator : EvaluationPlugin,
                 train_mb_size: int = 32,
                 train_epochs: int = 10,
                 eval_mb_size: Optional[int] = None,
                 device='cpu',
                 plugins: Optional[list[SupervisedPlugin]] = None,
                 eval_every: int = -1,
                 translator_only_epochs: int = 2,  # First phase of global training
                 joint_training_epochs: int = 8,   # Second phase of global training
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
        
        # Two-phase training configuration
        self.translator_only_epochs = translator_only_epochs
        self.joint_training_epochs = joint_training_epochs
        self.current_phase = "local"  # "local", "translator_only", "joint"
        
        # Initialize beta for annealing
        self.beta = 1.0

    def criterion(self):
        """Beta-VAE loss with annealing."""
        losses = self._criterion(self.mb_x, self.mb_output, beta=self.beta)
        return losses

    def _before_training_exp(self, **kwargs):
        """
        Called before training on each experience.
        PHASE 1: Local Training - Trigger model adaptation for new task.
        """
        super()._before_training_exp(**kwargs)
        
        # Trigger model adaptation for new task (creates local encoder + translator)
        if hasattr(self.model, 'adaptation'):
            self.model.adaptation(self.experience)
        
        # Start with local training phase
        self.current_phase = "local"

    def training_epoch(self, **kwargs):
        """
        Override training epoch to implement two-phase global training:
        1. Translator-only training (frozen global decoder)
        2. Joint training (translator + global decoder)
        """
        if self.clock.train_exp_epochs < self.translator_only_epochs:
            # PHASE 2a: Translator-only training with frozen decoder
            self.current_phase = "translator_only"
            self._freeze_global_decoder()
            super().training_epoch(**kwargs)
            
        elif self.clock.train_exp_epochs < (self.translator_only_epochs + self.joint_training_epochs):
            # PHASE 2b: Joint training (translator + global decoder)
            self.current_phase = "joint"
            self._unfreeze_global_decoder()
            super().training_epoch(**kwargs)
        else:
            # Continue joint training if more epochs specified
            self.current_phase = "joint"
            super().training_epoch(**kwargs)

    def _freeze_global_decoder(self):
        """Freeze global decoder parameters during translator-only training."""
        if hasattr(self.model, 'global_decoder'):
            for param in self.model.global_decoder.parameters():
                param.requires_grad = False

    def _unfreeze_global_decoder(self):
        """Unfreeze global decoder parameters during joint training."""
        if hasattr(self.model, 'global_decoder'):
            for param in self.model.global_decoder.parameters():
                param.requires_grad = True

    def make_train_dataloader(self,
                             num_workers=16,
                             shuffle=True,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             **kwargs):
        """Create training dataloader with proper configuration."""
        self.dataloader = DataLoader(
            self.adapted_dataset,
            shuffle=shuffle,
            batch_size=self.train_mb_size,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=kwargs.get("prefetch_factor", 8),
            num_workers=num_workers,
            drop_last=drop_last
        )

