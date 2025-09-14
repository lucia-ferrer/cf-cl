import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_float_to_uint8(images):
    """Convert float32 images in [0,1] range to uint8 in [0,255] range."""
    # Clamp to [0,1] to be safe
    images = torch.clamp(images, 0.0, 1.0)
    # Convert to [0,255] and uint8
    images = (images * 255).to(torch.uint8)
    return images