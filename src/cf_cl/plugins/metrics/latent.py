
from typing import Optional, Literal
from collections import defaultdict

num_cuda = None
import cupy as cp
try:
    num_cuda = cp.cuda.runtime.getDeviceCount()
except Exception as e:
    print(f"CUDA error, scikit-learn manifold methods. {e}")
    # Fallback to CPU libraries
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP
else:
    if num_cuda > 0:
        from cuml.manifold import TSNE, UMAP
        from cuml.decomposition import PCA
    else:
        # Fallback to CPU libraries if no GPU devices found
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from umap import UMAP

from cuml.metrics import trustworthiness

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

import torch

class TrustworthinessPluginMetric(PluginMetric[float]):
    """
    Plugin metric for computing trustworthiness score using cuML and CuPy arrays.
    """

    def __init__(self, n_neighbors=5, reset_at='experience', emit_at='experience', mode='eval'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.reset_at = reset_at
        self.emit_at = emit_at
        self.mode = mode
        self._original_data = []
        self._embedded_data = []

    def reset(self, **kwargs):
        self._original_data = []
        self._embedded_data = []

    def update(self, original_data, embedded_data):
        # Collect as CUDA tensors for later conversion
        self._original_data.append(original_data)
        self._embedded_data.append(embedded_data)

    def result(self, **kwargs) -> float:
        # Concatenate and convert to CuPy
        if self._original_data and self._embedded_data:
            full_original = torch.cat(self._original_data).cuda().detach()
            full_embedded = torch.cat(self._embedded_data).cuda().detach()
            # Convert to CuPy via DLPack
            score = trustworthiness(cp.asarray(full_original), cp.asarray(full_embedded),
                                    n_neighbors=self.n_neighbors)
            return score
        return 0.0

    # ========== EVALUATION HOOKS ==========
    def before_eval(self, strategy):
        if self.mode in ['eval', 'both'] and self.reset_at == 'stream':
            self.reset()

    def before_eval_exp(self, strategy):
        if self.mode in ['eval', 'both'] and self.reset_at == 'experience':
            self.reset()

    def before_eval_iteration(self, strategy):
        if self.mode in ['eval', 'both'] and self.reset_at == 'iteration':
            self.reset()

    def after_eval_iteration(self, strategy):
        if hasattr(strategy, 'mb_x') and hasattr(strategy, 'mb_output'):
            original_data = strategy.mb_x.cuda()
            *_, z = strategy.mb_output
            embedded_data = z.cuda()
            self.update(original_data, embedded_data)

    def after_eval_exp(self, strategy):
        if self.emit_at == 'experience':
            return self._package_result(strategy, phase='eval')
        return None

    def after_eval(self, strategy):
        if self.mode in ['eval', 'both'] and self.emit_at == 'stream':
            return self._package_result(strategy, phase='eval')
        return None

    def _package_result(self, strategy, phase='eval'):
        metric_value = self.result()
        if phase == 'eval' and self.emit_at == 'experience':
            plot_x_position = strategy.clock.train_exp_counter
        elif phase == 'eval' and self.emit_at == 'iteration':
            plot_x_position = getattr(strategy.clock, 'eval_iterations', 0)
        else:
            plot_x_position = strategy.clock.train_exp_counter
        metric_name = get_metric_name(
            self, strategy, add_experience=True, add_task=False
        )
        metric_name = f"{metric_name}/Eval"
        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "Trustworthiness"
    
    
class SampleLatentsPlugin(PluginMetric):
    """Metric used to sample latent representations and visualize them"""

    def __init__(
        self,
        *,
        mode: Literal["train", "eval", "both"] = "train",
        supervised: bool = True,
        manifold_embedding_method: Optional[Literal["tsne", "umap", "pca"]] = "umap",
        tag: str = "latent_space",
    ):
        super().__init__()
        self.mode = mode
        self.manifold_embedding_method = manifold_embedding_method
        self.supervised = supervised
        self._latent = None
        self.tag = f"{tag}_{manifold_embedding_method}"

    def after_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        if self.mode == "train" or self.mode == "both":
            return self._sample_latent_and_visualize(strategy)
        return None

    def after_eval_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        if self.mode == "eval" or self.mode == "both":
            return self._sample_latent_and_visualize(strategy)
        return None

    def reset(self) -> None:
        return None
    
    def result(self) -> torch.Tensor:
        return self._latent

    def _sample_latent_and_visualize(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        self._latent_from_samples(strategy)
        figure = self._plot_2d_latent(strategy)
        mt = self._to_metric_value(figure, strategy)
        return mt

    @torch.no_grad()
    def _latent_from_samples(self, strategy):
        """Extract latents from eval batch samples"""
        device = next(strategy.model.parameters()).device

        per_task_z = defaultdict(list[torch.Tensor])
        per_task_y = defaultdict(list[torch.Tensor])

        for exp in strategy.current_eval_stream:
            loader = DataLoader(
                exp.dataset,
                batch_size=strategy.eval_mb_size,
                shuffle=True,
                # generator=torch.Generator().manual_seed(42)  # optional for determinism
            )
            batch = next(iter(loader))
            if self.supervised:
                # expect (x, y, t); ignore extra fields
                x, y, t = batch
            else:
                # e.g., dataset yields (x, t); use task ids as pseudo-labels
                x, t = batch
                y = t

            x = x.to(device, non_blocking=True)
            y = y.to(device)
            t = t.to(device)

            with torch.no_grad():
                if hasattr(strategy.model, "encode"):
                    mu, var = strategy.model.encode(x)
                    z = strategy.model.reparameterize(mu, var)
                    
                else:
                    task_id = strategy.experience.current_experience
                    # task_id = t.unique()
                    z = strategy.model.get_global_latent_representations(x, task_id)
                    
            # 1D long labels for masking
            y1d = y.view(-1).long()
            classes = torch.unique(y1d)

            for cls_ in classes.tolist():
                mask = (y1d == cls_)
                per_task_z[cls_].append(z[mask])
                per_task_y[cls_].append(y1d[mask])

        # Concatenate per class on the same device
        per_task_z_tensor = defaultdict(torch.Tensor)
        per_task_y_tensor = defaultdict(torch.Tensor)
        for cls_ in list(per_task_z.keys()):
            per_task_z_tensor[cls_] = torch.cat(per_task_z[cls_], dim=0)
            per_task_y_tensor[cls_] = torch.cat(per_task_y[cls_], dim=0)
        
        self._latent_tensors = per_task_z_tensor
        self._labels_tensors = per_task_y_tensor
            
    def _plot_2d_latent(self, strategy):
        # Extract latents and labels
        z = torch.cat(list(self._latent_tensors.values())).detach()
        if str(strategy.device) == 'cpu':
            import numpy as np
            z = np.asarray(z)
        else:    
            z = cp.asarray(z)
        if z.shape[1] == 2 : 
            self.manifold_embedding_method = None
        # Perform PCA or t-SNE
        match self.manifold_embedding_method:
            case "tsne":
                tsne = TSNE(n_components=2, random_state=42)
                z_embedded = tsne.fit_transform(z)
            case "pca":
                pca = PCA(n_components=2)
                z_embedded = pca.fit_transform(z)       
            case "umap":
                umap = UMAP(n_components=2)
                z_embedded = umap.fit_transform(z)
            case _:
                z_embedded = z[:, :2]
        
        if isinstance(z_embedded, cp.ndarray):
            z_embedded = cp.asnumpy(z_embedded)

        # Set colors by class or task 
        if self.supervised:
            y = torch.cat(list(self._labels_tensors.values()))
            c = y.detach().cpu().numpy()
            base = plt.get_cmap("tab20")
            cmap_lbl = "Class"
            names = torch.unique(y).tolist()
        else: 
            raise NotImplementedError
        
        colors = [base(i / len(names)) for i in range(len(names))]
        cmap = ListedColormap(colors)
        
        # Build graph 
        plt.figure()
        sc = plt.scatter(
            z_embedded[:,0], z_embedded[:, 1],
            s=10,
            c=c,
            cmap=cmap,
            alpha=0.7, 
        )
        plt.colorbar(sc, label=cmap_lbl)
        
        return plt.gcf()
    
    def _to_metric_value(self, fig: plt.Figure, strategy) -> MetricValue:
        cur_exp = strategy.experience
        exp_id = getattr(cur_exp, "current_experience", 0)
        name = f"{self.tag}/exp{exp_id}"
        return MetricValue(
            self,
            name=name,
            value=fig,
            x_plot=strategy.clock.train_iterations,
        )

if num_cuda is not None:    
    latent_plugs = [SampleLatentsPlugin(mode="eval",manifold_embedding_method="umap"),
                    TrustworthinessPluginMetric(mode="eval")
    ]
else:
    latent_plugs = [SampleLatentsPlugin(mode="eval",manifold_embedding_method="umap")]