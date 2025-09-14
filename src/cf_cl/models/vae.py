import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConvEncoder(nn.Module):
    """
    Dynamic Encoder with convolutions that adapt based on image size and channels.
    """
    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim: int = 64,
                 base_ch: int = 64,
                 ):
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

        # VAE latent outputs
        self.fc_mu = nn.Linear(fc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_out_dim, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class DynamicConvDecoder(nn.Module):
    """
    Dynamic Decoder with convolutions that adapt based on target image size.
    """
    def __init__(self, latent_dim: int, features_shape: tuple, img_channels: int, img_size: int, base_ch: int = 32, final_activation:str|None=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.features_shape = features_shape  # (C, H, W)
        self.img_channels = img_channels
        self.img_size = img_size
        self.base_ch = base_ch

        C0, H0, W0 = features_shape

        # Dynamic FC based on image size
        if img_size < 32 and img_channels == 1:
            self.fc1 = nn.Linear(latent_dim, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, C0 * H0 * W0)
        else:
            self.fc1 = nn.Linear(latent_dim, 512)
            self.fc2 = nn.Linear(512, 1024)
            self.fc3 = nn.Linear(1024, C0 * H0 * W0)

        # Calculate upsampling stages needed from (H0, W0) to (img_size, img_size)
        scale_factor = img_size / H0
        stages_needed = int(math.log2(scale_factor)) if scale_factor > 1 else 0

        # Build upsampling layers dynamically
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
        self.final_activation = final_activation

    def forward(self, z):
        batch_size = z.size(0)

        # MLP expansion
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Reshape to feature map
        x = x.view(batch_size, *self.features_shape)
        x = self.decoder(x)

        # Ensure exact target dimensions
        _, _, H, W = x.shape
        if H != self.target_size[0] or W != self.target_size[1]:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        if self.final_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_activation == 'tanh':
            x = torch.tanh(x)
        return x


class VAE(nn.Module):
    """
    Simple VAE using the dynamic convolution architecture.
    """
    def __init__(self,
                 img_channels: int = 3,
                 img_size: int = 32,
                 latent_dim: int = 64,
                 base_ch: int = 64,
                 activation: str|None = 'sigmoid',
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.base_ch = base_ch
        self.device = device

        # Dynamic encoder
        self.encoder = DynamicConvEncoder(
            in_channels=img_channels,
            img_size=img_size,
            latent_dim=latent_dim,
            base_ch=base_ch        )

        # Dynamic decoder - get features_shape from encoder
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, img_size, img_size)
            _ = self.encoder.conv_layers(dummy)
        features_shape = self.encoder.features_shape

        self.decoder = DynamicConvDecoder(
            latent_dim=latent_dim,
            features_shape=features_shape,
            img_channels=img_channels,
            img_size=img_size,
            base_ch=base_ch,
            final_activation=activation
        )

        self.to(self.device)

    def encode(self, x):
        x = x.to(self.device)
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for continuous variables."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.to(self.device)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    @torch.no_grad()
    def generate(self, x):
        x = x.to(self.device)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)


# Loss function (same as original)
def vae_loss(x, forward_output, beta=1.0):
    """VAE loss function compatible with VAE output."""
    x_recon, mu, logvar, z = forward_output

    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss normalized by batch size
    batch_size = x.size(0)
    total_loss = (recon_loss + beta * kl_loss) / batch_size

    return total_loss


# Loss function (same as original)
def vae_bce_loss(x, forward_output, beta=1.0):
    """VAE loss function compatible with VAE output."""
    x_recon, mu, logvar, z = forward_output

    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss normalized by batch size
    batch_size = x.size(0)
    total_loss = (recon_loss + beta * kl_loss) / batch_size

    return total_loss