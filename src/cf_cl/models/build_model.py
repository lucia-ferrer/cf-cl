import torch 
from torch.optim import SGD, Adam
from .vae import VAE, vae_loss, vae_bce_loss
from .multiband_vae import MultiBandVAE, multiband_vae_loss

def build_model_optim(model_cfg):
    """Based on Config will build Experiment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_cfg.arch == "CnnVAE":
        model = VAE(
            img_channels=model_cfg.img_channels,
            img_size=model_cfg.img_size,
            latent_dim=model_cfg.latent_dim,
            activation=model_cfg.activation,
            device=device
        )
        criterion = vae_bce_loss if model_cfg.activation == "sigmoid" else vae_loss

    else: 
        raise NotImplementedError

    if model_cfg.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=model_cfg.lr)
    else:
        optimizer = SGD(model.parameters(), lr=model_cfg.lr)

    
    return model, optimizer, criterion


def build_multiband_model(model_cfg):
    """ Build Experiment for Multiband VAE"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiBandVAE(
        img_channels=model_cfg.img_channels,
        img_size=model_cfg.img_size,
        local_latent_dim=model_cfg.latent_dim,
        categorical_dim=model_cfg.categorical_dim,
        global_latent_dim=model_cfg.latent_dim,
    )
    model.to(device)
    criterion = multiband_vae_loss
    optimizer = Adam(model.parameters(), lr=model_cfg.lr)
    return model, optimizer, criterion
    
    