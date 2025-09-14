import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    def __init__(
        self,
        img_channels=1,
        img_size=28,
        latent_dim=64,
        base_ch=32,
        base_fc=256,
        dropout=0.2,
        dec_activation="sigmoid",
        device="cpu",
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = torch.device(device)

        # Encoder: 
        self.enc_conv = nn.Sequential(
            nn.Conv2d(
                img_channels, base_ch, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # Add spatial dropout
            nn.Conv2d(
                base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # Add spatial dropout
            
        )
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, img_size, img_size)
            h = self.enc_conv(dummy)
            self._feat_shape = h.shape[1:]  # (C, H, W) = (base_ch*2, 7, 7)
            self.flat_dim = h.numel()

        self.enc_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, base_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(base_fc, base_fc // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(base_fc // 2, latent_dim)
        self.fc_logvar = nn.Linear(base_fc // 2, latent_dim)

        # Decoder MLP
        self.dec_mlp = nn.Sequential(
            nn.Linear(latent_dim, base_fc//2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),  # Light regularization

            nn.Linear(base_fc//2, base_fc),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.05),  # Light regularization
            
            nn.Linear(base_fc, self.flat_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(
                base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d( 
                base_ch, base_ch, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(base_ch, img_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.dec_upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # nearest often works better than bilinear
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(inplace=True),
            
            # Final layer
            nn.Conv2d(base_ch, img_channels, kernel_size=3, padding=1)
        )
        
        if dec_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif dec_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

        self.to(self.device, non_blocking=True)

    @property
    def feat_shape(self):
        return self._feat_shape

    def encode(self, x):
        x = x.to(self.device, non_blocking=True)
        h = self.enc_conv(x)
        h = self.enc_mlp(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.to(self.device, non_blocking=True)
        h = self.dec_mlp(z)
        h = h.view(-1, *self.feat_shape)
        x_recon_logits = self.dec_conv(h)
        return self.out_act(x_recon_logits)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    @torch.no_grad()
    def generate(self, x):
        x = x.to(self.device, non_blocking=True)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)
    

BCE_loss = nn.BCELoss(reduction="sum")
MSE_loss = nn.MSELoss(reduction="sum")

def vae_loss(x, forward_output, beta=1.0):
    x_hat, mu, logvar = forward_output
    recon_loss = MSE_loss(x_hat, x) / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + beta * kl
    
    return total_loss

def vae_bce_loss(x, forward_output, beta=1.0):
    x_hat, mu, logvar = forward_output
    recon_loss = BCE_loss(x_hat, x) / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + beta * kl
    
    return total_loss