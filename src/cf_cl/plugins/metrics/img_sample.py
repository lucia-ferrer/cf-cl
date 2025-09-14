from collections import defaultdict
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, TensorImage
from avalanche.evaluation.metrics import ImagesSamplePlugin


class ReconImagesSamplePlugin(ImagesSamplePlugin):
    """
    Two-row image grid:
      - Top row: input images
      - Bottom row: model reconstructions (or predictions)
    Works wherever ImagesSamplePlugin works. It computes predictions once
    per sampling using strategy.model in eval/no_grad.
    """

    def __init__(self, n_rows: int = 4, n_cols: int = 8, **kwargs):
        # Keep a "base" number of rows for inputs only
        self.base_rows = n_rows
        # Ask the base plugin to render double rows (inputs + outputs)
        super().__init__(n_rows=2 * n_rows, n_cols=n_cols, **kwargs)

        # Cache for convenience
        self._base_wanted = n_cols * self.base_rows  # inputs count

    def _predict(self, model, x: torch.Tensor) -> torch.Tensor:
        # Inference-only forward pass that returns images shaped like inputs
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                out = model(x)
                # If the model returns a tuple like (x_recon, mu, logvar), take first
                if isinstance(out, (list, tuple)):
                    out = out[0]
        finally:
            # Restore mode
            model.train(was_training)
        return out

    def _load_data(self, strategy) -> tuple[list[torch.Tensor], list[int], list[int]]:
        """
        Collect exactly base_wanted inputs, run one forward pass to get
        outputs, then concatenate inputs + outputs to make 2 * base_wanted
        images. Labels/tasks are duplicated to match length.
        """
        assert strategy.adapted_dataset is not None

        device = next(strategy.model.parameters()).device
        dataloader = self._make_dataloader(
            strategy.adapted_dataset, strategy.eval_mb_size
        )

        inputs: list[torch.Tensor] = []
        labels: list[int] = []
        tasks: list[int] = []

        # Collect exactly N inputs (no recon here)
        for batch_images, batch_labels, batch_tasks in dataloader:
            # Ensure tensors are in standard list-of-tensors shape
            need = self._base_wanted - len(inputs)
            if need <= 0:
                break

            # Slice current batch
            cur_x = batch_images[:need]
            cur_y = batch_labels[:need]
            cur_t = batch_tasks[:need]

            # Extend inputs/labels/tasks
            # Note: batch_images may be a tensor [B,C,H,W]; turn into list of tensors
            if isinstance(cur_x, torch.Tensor):
                inputs.extend(list(cur_x))
            else:
                inputs.extend(cur_x)
            if hasattr(cur_y, "tolist"):
                labels.extend(cur_y.tolist())
            else:
                labels.extend(list(cur_y))
            if hasattr(cur_t, "tolist"):
                tasks.extend(cur_t.tolist())
            else:
                tasks.extend(list(cur_t))

            if len(inputs) >= self._base_wanted:
                break

        else:
            return [], [], []

        x_stack = torch.stack(inputs, dim=0).to(device, non_blocking=True)
        y_stack = self._predict(strategy.model, x_stack)
        y_stack = y_stack.detach().cpu()
        x_cpu = x_stack.detach().cpu()

        # Concatenate: first all inputs, then all outputs
        full_images_tensor = torch.cat([x_cpu, y_stack], dim=0)

        # Convert back to list of tensors for the base plugin API
        images: list[torch.Tensor] = list(full_images_tensor)

        # Duplicate labels/tasks to match images length (inputs + outputs)
        full_labels = labels + labels
        full_tasks = tasks + tasks

        # The base plugin will take exactly n_rows * n_cols images,
        # which we configured as 2 * base_rows * n_cols
        return images, full_labels, full_tasks


class ReconPerClassMetric(PluginMetric):
    """
    Avalanche-centric image metric for TensorBoard:
      - Emits two images at each eval experience: past and future.
      - Each image is a 2-row grid (inputs / reconstructions), 1 column per class.
    Relies on strategy.scenario streams and strategy.make_eval_dataloader.
    """

    def __init__(
        self,
        mode: Literal["eval", "both"] = "eval",
        max_classes: Optional[int] = None,
        tag: str = "recon_per_class",
    ):
        super().__init__()
        self.mode = mode
        self.max_classes = max_classes
        self.tag = tag
        self._current_images: dict[str, Optional[torch.Tensor]] = {
            "past": None,
            "future": None,
        }

    # PluginMetric abstract methods
    def result(self):
        img = self._current_images.get("past")
        return None if img is None else TensorImage(img)

    def reset(self):
        self._current_images = {"past": None, "future": None}

    # Eval-time hook
    def after_eval_dataset_adaptation(self, strategy):
        if self.mode in ("eval", "both"):
            return self._compute_and_emit(strategy)
        return None

    # Optional train-time hook if desired
    def after_train_dataset_adaptation(self, strategy):
        if self.mode == "both":
            return self._compute_and_emit(strategy)
        return None

    def _compute_and_emit(self, strategy):

        # Collect one sample per class from past and future using Avalanche loaders
        per_class = self._one_per_class_from_stream(strategy)

        imgs = self._build_grid_and_reconstruct(strategy, per_class)

        mvs: MetricValue = self._to_metric_value(imgs, strategy)

        return mvs

    def _one_per_class_from_stream(self, strategy) -> dict[int, torch.Tensor]:
        per_class: dict[int, torch.Tensor] = defaultdict(lambda: torch.empty(0))

        for exp in strategy.current_eval_stream:
            exp.eval()
            loader = DataLoader(exp.dataset, shuffle=True)
            tasks = []
            _classes = exp.classes_in_this_experience
            n_classes = len(_classes)
            x, y, t = next(iter(loader))
            per_class[y.item()] = x
            tasks.extend(t)

        return per_class

    def _build_grid_and_reconstruct(
        self, strategy, per_class: dict[int, torch.Tensor]
    ) -> torch.Tensor:
        # Order by class id for stable columns
        labels = sorted(per_class.keys())
        xs = [per_class[y] for y in labels]

        device = next(strategy.model.parameters()).device
        x_stack = torch.cat(xs, dim=0).to(device, non_blocking=True)

        # Single forward in eval/no_grad; take first output if model returns tuple
        was_training = strategy.model.training
        try:
            strategy.model.eval()
            with torch.no_grad():
                out = strategy.model(x_stack)
                if isinstance(out, (list, tuple)):
                    out = out[0]
        finally:
            strategy.model.train(was_training)

        x_cpu = x_stack.detach().cpu()
        y_cpu = out.detach().cpu()

        if x_cpu.dim() == 3:
            x_cpu = x_cpu.unsqueeze(0)
            y_cpu = y_cpu.unsqueeze(0)

        N = len(labels)
        # Build 2-row grid by horizontal concatenation of columns
        top_row = torch.cat([x_cpu[i] for i in range(N)], dim=-1)  # C,H,N*W
        bot_row = torch.cat([y_cpu[i] for i in range(N)], dim=-1)  # C,H,N*W
        img = torch.cat([top_row, bot_row], dim=-2)  # C,2H,N*W
        return img

    def _to_metric_value(self, img: torch.Tensor, strategy) -> MetricValue:
        cur_exp = strategy.experience
        exp_id = getattr(cur_exp, "current_experience", 0)
        name = f"{self.tag}/exp{exp_id}"
        return MetricValue(
            self,
            name=name,
            value=TensorImage(img),
            x_plot=strategy.clock.train_iterations,
        )


recon_imgs = ReconPerClassMetric()
img_plugs = ReconImagesSamplePlugin(n_cols=5, n_rows=1, group=False, mode="eval")

visual_metrics = [recon_imgs, img_plugs]