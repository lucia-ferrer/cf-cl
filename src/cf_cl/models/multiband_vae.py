from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import copy
from avalanche.models import DynamicModule
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import EvaluationPlugin


class ConvEncoder(nn.Module):
    """
    Local Encoder with continuous latent (VAE) and optional categorical latent space.
    Supports temperature annealing for Gumbel-Softmax.
    """
    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 32,
                 latent_dim: int = 64,
                 categorical_dim: int = 0,
                 base_ch: int = 64,
                 temperature: float = 1.0):
        super().__init__()
        
        # Determine number of conv blocks based on image size and channels
        if img_size < 32 and in_channels == 1:
            self.num_conv_blocks = 2
            self.hidden_ch = base_ch * 2  # Reduced final channels
        else:
            self.num_conv_blocks = 4
            self.hidden_ch = base_ch * 8
        
        # Build conv layers dynamically
        conv_layers = []
        curr_ch = in_channels
        
        for i in range(self.num_conv_blocks):
            if i == self.num_conv_blocks - 1:
                out_ch = self.hidden_ch
            else:
                out_ch = base_ch * (2 ** i)
            
            conv_layers.extend([
                nn.Conv2d(curr_ch, out_ch, 4, 2, 1),
                nn.ReLU(True)
            ])
            curr_ch = out_ch
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate output dimensions after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            h = self.conv_layers(dummy)
        self.features_shape = h.shape[1:]  # (C, H, W)
        self.flatten_hidden = h.numel()
        
        self.flatten = nn.Flatten()
        
        # Adaptive FC layers based on architecture depth
        if self.num_conv_blocks == 2:
            self.fc1 = nn.Linear(self.flatten_hidden, 256)
            self.fc2 = nn.Linear(256, 128)
            fc_out_dim = 128
        else:
            self.fc1 = nn.Linear(self.flatten_hidden, 1024)
            self.fc2 = nn.Linear(1024, 512)
            fc_out_dim = 512
        
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        if categorical_dim == 0:
            # Only continuous latent
            self.fc_mu = nn.Linear(fc_out_dim, latent_dim)
            self.fc_logvar = nn.Linear(fc_out_dim, latent_dim)
        else:
            # Both continuous and categorical latent
            self.fc_mu = nn.Linear(fc_out_dim, latent_dim)
            self.fc_logvar = nn.Linear(fc_out_dim, latent_dim)
            self.fc_cat = nn.Linear(fc_out_dim, latent_dim * categorical_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Continuous latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        if self.categorical_dim == 0:
            return mu, logvar, None
        else:
            # Categorical latent space with Gumbel-Softmax
            cat_logits = self.fc_cat(x).view(-1, self.latent_dim, self.categorical_dim)
            q_y_soft = F.gumbel_softmax(cat_logits, tau=self.temperature, hard=False, dim=-1)
            return mu, logvar, q_y_soft



class Translator(nn.Module):
    """
    Maps (local latent, task id) to global latent space.
    Handles both continuous and categorical latents.
    """
    def __init__(self, local_latent_dim:int, categorical_dim:int, global_latent_dim:int, num_tasks:int):
        super().__init__()
        self.local_latent_dim = local_latent_dim
        self.categorical_dim = categorical_dim
        
        # Calculate input dimension: continuous + categorical (flattened) + task one-hot
        cat_dim = local_latent_dim * categorical_dim if categorical_dim > 0 else 0
        self.input_dim = local_latent_dim + cat_dim + num_tasks
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, global_latent_dim)
        )

    def forward(self, mu, q_y, task_id):
        batch_size = mu.size(0)
        device = mu.device
        
        # Convert task_id to one-hot if it's an integer
        if isinstance(task_id, int):
            num_tasks = self.input_dim - mu.size(1) - (mu.size(1) * self.categorical_dim if q_y is not None else 0)
            task_id_onehot = torch.zeros(batch_size, num_tasks, device=device)
            task_id_onehot[:, task_id] = 1
        else:
            task_id_onehot = task_id.to(device)
        
        # Concatenate continuous and categorical (flattened) with task embedding
        if q_y is None:
            input_tensor = torch.cat([mu, task_id_onehot], dim=1)
        else:
            input_tensor = torch.cat([mu, q_y.view(batch_size, -1), task_id_onehot], dim=1)
        
        return self.network(input_tensor)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim:int, features_shape:tuple[int, int, int], img_channels:int, img_size:int, base_ch:int=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.features_shape = features_shape  # (C, H, W)
        self.img_channels = img_channels
        self.img_size = img_size
        self.base_ch = base_ch
        
        C0, H0, W0 = features_shape
        
        # Dynamic FC
        if img_size < 32 and img_channels == 1:
            self.fc1 = nn.Linear(latent_dim, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, C0 * H0 * W0)
        else:
            self.fc1 = nn.Linear(latent_dim, 512)
            self.fc2 = nn.Linear(512, 1024)
            self.fc3 = nn.Linear(1024, C0 * H0 * W0)
        
        #  from (H0, W0) to (img_size, img_size)
        scale_factor = img_size / H0
        stages_needed = int(math.log2(scale_factor)) if scale_factor > 1 else 0
        
        # Build upsampling layers
        deconv_layers = []
        in_ch = C0
        current_h, current_w = H0, W0
        
        for i in range(stages_needed):
            # Calculate output channels for this stage
            if i == stages_needed - 1:
                # Last transposed conv - output final image channels
                out_ch = img_channels
            else:
                # Progressive channel reduction
                reduction_factor = 2 ** (stages_needed - i - 1)
                out_ch = max(base_ch // reduction_factor, base_ch // 4)
            
            # Calculate exact target dimensions for this stage
            if i == stages_needed - 1:
                # Final stage: reach exact target size
                target_h = img_size
                target_w = img_size
            else:
                # Intermediate stage: double current size
                target_h = current_h * 2
                target_w = current_w * 2
            
            # Calculate output_padding needed for exact dimensions
            # Formula: output = (input - 1) * stride - 2 * padding + kernel_size + output_padding
            kernel_size = 4
            stride = 2
            padding = 1
            
            expected_h = (current_h - 1) * stride - 2 * padding + kernel_size
            expected_w = (current_w - 1) * stride - 2 * padding + kernel_size
            
            output_padding_h = target_h - expected_h
            output_padding_w = target_w - expected_w
            
            # Ensure output_padding is valid (0 <= output_padding < stride)
            output_padding_h = max(0, min(output_padding_h, stride - 1))
            output_padding_w = max(0, min(output_padding_w, stride - 1))
            
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch, 
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=padding,
                    output_padding=(output_padding_h, output_padding_w)
                )
            )
            
            # Add activation (except for final layer)
            if i < stages_needed - 1:
                deconv_layers.append(nn.LeakyReLU(inplace=True))
            
            # Update for next iteration
            in_ch = out_ch
            current_h = target_h
            current_w = target_w
        
        # If no upsampling needed or not enough stages, add final conv
        if stages_needed == 0 or current_h != img_size or current_w != img_size:
            deconv_layers.extend([
                nn.Conv2d(in_ch, img_channels, kernel_size=3, padding=1),
            ])
        
        self.decoder = nn.Sequential(*deconv_layers)
        self.target_size = (img_size, img_size)

    def forward(self, z):
        batch_size = z.size(0)
        
        # MLP expansion
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Reshape to feature map
        x = x.view(batch_size, *self.features_shape)
        x = self.decoder(x)
        
        # Exact target dimensions
        _, _, H, W = x.shape
        if H != self.target_size[0] or W != self.target_size[1]:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(x)


class MultiBandVAE(DynamicModule):
    """
    MultiBand VAE implementation for continual learning.
    Maintains separate local encoders for each task and aligns them through translators.
    """
    def __init__(self,
                 img_channels: int = 3,
                 img_size: int = 32,
                 local_latent_dim: int = 64,
                 categorical_dim: int = 0,
                 global_latent_dim: int = 64,
                 base_ch: int = 64,
                 ):
        super().__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.local_latent_dim = local_latent_dim
        self.categorical_dim = categorical_dim
        self.global_latent_dim = global_latent_dim
        self.base_ch = base_ch
        
        # Temperature annealing parameters
        self.temperature = 1.0
        self.min_temperature = 0.5
        self.anneal_rate = 0.00003
        
        # Dynamic components for each task
        self.local_encoders = nn.ModuleList()
        self.translators = nn.ModuleList()
        
        # Global decoder (shared across all tasks)
        dummy_encoder = ConvEncoder(
            in_channels=self.img_channels,
            img_size=self.img_size,
            latent_dim=self.local_latent_dim,
            categorical_dim=self.categorical_dim,
            base_ch=self.base_ch,
            temperature=self.temperature
        )
        features_shape = dummy_encoder.features_shape
        
        self.global_decoder = ConvDecoder(
            latent_dim=self.global_latent_dim,
            features_shape=features_shape,
            img_channels=self.img_channels,
            base_ch=self.base_ch,
            img_size=self.img_size
        )
        
        # Store frozen models for replay
        self.frozen_translators = []
        self.frozen_decoder = None
        self.current_task = 0

    def adaptation(self, experience):
        """Adapt before experience - create Local Encoder + Translator (PHASE 1)"""
        # Create new local encoder
        new_encoder = ConvEncoder(
            in_channels=self.img_channels,
            img_size=self.img_size,
            latent_dim=self.local_latent_dim,
            categorical_dim=self.categorical_dim,
            base_ch=self.base_ch,
            temperature=self.temperature
        )
        
        # Move to same device as model
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        new_encoder = new_encoder.to(device)
        
        self.local_encoders.append(new_encoder)
        
        # Create new translator
        new_translator = Translator(
            self.local_latent_dim,
            self.categorical_dim,
            self.global_latent_dim,
            self.current_task + 1
        )
        
        # Move to same device as model
        new_translator = new_translator.to(device)
        self.translators.append(new_translator)
        
        # Store frozen copies for replay (if not first task)
        if self.current_task > 0:
            # FIXED: Use -2 to get the previous translator, not the current one
            frozen_translator = copy.deepcopy(self.translators[-2])
            frozen_translator.eval()
            for param in frozen_translator.parameters():
                param.requires_grad = False
            self.frozen_translators.append(frozen_translator)
            
            if self.frozen_decoder is None:
                self.frozen_decoder = copy.deepcopy(self.global_decoder)
                self.frozen_decoder.eval()
                for param in self.frozen_decoder.parameters():
                    param.requires_grad = False
        
        self.current_task += 1

    def update_temperature(self):
        """Anneal temperature during training."""
        decay_factor = torch.exp(torch.tensor(-self.anneal_rate)).item()
        self.temperature = max(self.min_temperature, self.temperature * decay_factor)
        
        # Update temperature in all encoders
        for encoder in self.local_encoders:
            encoder.temperature = self.temperature

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for continuous variables."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_current_task_data(self, x, task_id):
        """Encode data using the appropriate local encoder."""
        mu, logvar, q_y = self.local_encoders[task_id](x)
        z_continuous = self.reparameterize(mu, logvar)
        return mu, logvar, q_y, z_continuous

    def get_global_latent_representations(self, x, task_id):
        """Get global latent representations for current task data."""
        mu, logvar, q_y, z_continuous = self.encode_current_task_data(x, task_id)
        z_global = self.translators[task_id](z_continuous, q_y, task_id)
        return z_global

    def generate_replay_samples(self, num_samples: int, task_id: int):
        """Generate replay samples for a specific previous task."""
        with torch.no_grad():
            # Get device from model parameters
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            
            # Sample from local latent space on correct device
            lambda_samples = torch.randn(num_samples, self.local_latent_dim, device=device)
            
            if task_id < len(self.frozen_translators):
                # Use frozen translator for previous tasks
                z_global = self.frozen_translators[task_id](lambda_samples, None, task_id)
                # Use frozen decoder if available, otherwise current decoder
                decoder = self.frozen_decoder if self.frozen_decoder is not None else self.global_decoder
                x_replay = decoder(z_global)
                return x_replay, lambda_samples
            
        return None, None

    def forward(self, x, task_labels: Optional[torch.Tensor] = None):
        """Forward pass with current task data."""
        # Get device from model parameters
        device = next(self.parameters()).device if list(self.parameters()) else x.device
        
        # Ensure input is on correct device
        x = x.to(device)
        
        if task_labels is None:
            # Create task labels on correct device
            task_labels = torch.tensor([self.current_task - 1] * x.size(0), device=device)
        else:
            task_labels = task_labels.to(device)
            
        if len(task_labels.shape) == 0:
            task_labels = task_labels.unsqueeze(0)
        
        task_id = int(task_labels[0].item())
        
        # Validate task_id
        if task_id < 0 or task_id >= len(self.local_encoders):
            raise ValueError(f"Invalid task_id {task_id}. Available encoders: {len(self.local_encoders)}")
        
        # Encode with appropriate local encoder
        mu, logvar, q_y = self.local_encoders[task_id](x)
        
        # Reparameterize continuous variables
        z_continuous = self.reparameterize(mu, logvar)
        
        # Translate to global space
        z_global = self.translators[task_id](z_continuous, q_y, task_id)
        
        # Decode
        x_recon = self.global_decoder(z_global)
        
        return x_recon, mu, logvar, q_y, z_global

    def to(self, device):
        """Override to method to ensure all components are moved to device."""
        super().to(device)
        
        # Move frozen components if they exist
        for frozen_translator in self.frozen_translators:
            frozen_translator.to(device)
        
        if self.frozen_decoder is not None:
            self.frozen_decoder.to(device)
        
        return self

def multiband_vae_loss(targets, predictions, beta=1.0, **kwargs):
    """
    Enhanced loss function supporting both continuous and categorical latents.
    """
    x_recon, mu, logvar, q_y, z_global = predictions
    recon_loss = F.binary_cross_entropy(x_recon, targets, reduction='sum') #mse_loss
    kl_continuous = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_categorical = 0
    if q_y is not None:
        # KL divergence between q_y and uniform prior
        batch_size, latent_dim, cat_dim = q_y.shape
        uniform_prior = torch.ones_like(q_y) / cat_dim
        # KL(q||p) = sum(q * log(q/p))
        log_ratio = torch.log(q_y + 1e-20) - torch.log(uniform_prior + 1e-20)
        kl_categorical = torch.sum(q_y * log_ratio)
    
    # Total loss
    batch_size = targets.size(0)
    total_loss = (recon_loss + beta * (kl_continuous + kl_categorical)) / batch_size
    
    return total_loss