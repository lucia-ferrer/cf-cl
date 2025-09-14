
from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation import Metric
from torchmetrics.image.fid import FrechetInceptionDistance
from avalanche.evaluation.metric_utils import get_metric_name
import torch


from vae_ls_align.utils import convert_float_to_uint8


class FIDStandaloneMetric(Metric[float]):
    """Standalone FID metric wrapping torchmetrics FID"""
    
    def __init__(self, feature=2048, reset_real_features=True, normalize=False, device='cuda'):
        super().__init__()
        self._has_real_images = False
        self.device = device if device is not None else torch.device('cpu')
        self._fid_metric = FrechetInceptionDistance(
            feature=feature,
            reset_real_features=reset_real_features,
            normalize=normalize
        ).to(self.device)
    
    def update(self, real_images=None, fake_images=None):
        if real_images is not None:
            self._fid_metric.update(real_images, real=True)
            self._has_real_images = True
        if fake_images is not None:
            self._fid_metric.update(fake_images, real=False)
    
    def result(self):
        if not self._has_real_images:
            return float('nan')
        try:
            fid_value = self._fid_metric.compute()
            return fid_value.item() if torch.is_tensor(fid_value) else float(fid_value)
        except Exception as e:
            print(f"Error computing FID: {e}")
            return float('nan')
    
    def reset(self):
        self._fid_metric.reset()
        self._has_real_images = False


class FIDPluginMetric(GenericPluginMetric):
    """
    Fréchet Inception Distance (FID) metric for Avalanche EvalPlugin.
    """
    
    def __init__(self, 
                 feature=64, 
                 reset_at='experience', 
                 emit_at='experience', 
                 mode='eval',
                 reset_real_features=True,
                 normalize=False,
                 device='cuda'):
        # Create the standalone metric
        fid_metric = FIDStandaloneMetric(
            feature=feature,
            reset_real_features=reset_real_features,
            normalize=normalize,
            device=device
        )
        
        # Initialize GenericPluginMetric with the standalone metric
        super().__init__(
            metric=fid_metric,
            reset_at=reset_at, 
            emit_at=emit_at, 
            mode=mode
        )

    def update(self, strategy):
        """Update the metric with real and generated images."""
        real_images = None
        fake_images = None
        
        # Update with real images (adjust based on your use case)
        if hasattr(strategy, 'mb_x') and strategy.mb_x is not None:
            real_images = convert_float_to_uint8(strategy.mb_x).to(self._metric.device)
        
        # Update with generated images (adjust based on your use case) 
        if hasattr(strategy, 'mb_output') and strategy.mb_output is not None:
            x_reco, *_, z = strategy.mb_output
            fake_images = convert_float_to_uint8(x_reco).to(self._metric.device)
        
        assert torch.is_tensor(real_images) and real_images.dtype == torch.uint8, f"Image is {type(real_images)} with dtype={real_images.dtype}"
        assert torch.is_tensor(fake_images) and fake_images.dtype == torch.uint8, f"Image is {type(fake_images)} with dtype={fake_images.dtype}"
        self._metric.update(real_images=real_images, fake_images=fake_images)

    def __str__(self):
        return "FID"


def image_quality_metrics(epoch=False,
                          epoch_running=False,
                          experience=True,
                          stream=False,
                          ):
    """Generates Avalache Plugin for Evalute, whose metric measure the reconstruction of the image

    Args:
        epoch (bool, optional):  If True, will return a metric able to log the epoch. Defaults to False.
        epoch_running (bool, optional): _description_. Defaults to False.
        experience (bool, optional):  If True, will return a metric able to log the experience. Defaults to False.
        stream (bool, optional):  If True, will return a metric able to log the stram. Defaults to False.
        data_range (float, optional): Range for the data image input, for the PSNR metric. Defaults to 2.0, as 
            the PSNR metric uses [0,1] as range.
        
    Returns:
        list[PluginMetric]: list of metrics to pass directly to the evaluator
    """
    metrics: list = [FIDPluginMetric(feature=64,
                                       reset_at="epoch" if epoch or epoch_running else "experience",
                                       emit_at="epoch" if epoch or epoch_running else "experience",
                                       mode="eval")
    ]

    return metrics