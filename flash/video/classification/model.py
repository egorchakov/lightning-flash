# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import FunctionType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from torchmetrics import Accuracy

from flash.core.classification import ClassificationTask
from flash.core.registry import FlashRegistry
from flash.utils.imports import _PYTORCHVIDEO_AVAILABLE

_VIDEO_CLASSIFIER_MODELS = FlashRegistry("backbones")

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.models import hub
    for fn_name in dir(hub):
        if "__" not in fn_name:
            fn = getattr(hub, fn_name)
            if isinstance(fn, FunctionType):
                _VIDEO_CLASSIFIER_MODELS(fn=fn)


class VideoClassifierFinetuning(BaseFinetuning):

    def __init__(self, num_layers: int = 5, train_bn: bool = True, unfreeze_epoch: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.train_bn = train_bn
        self.unfreeze_epoch = unfreeze_epoch

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        self.freeze(modules=list(pl_module.model.children())[:-self.num_layers], train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        if epoch != self.unfreeze_epoch:
            return
        self.unfreeze_and_add_param_group(
            modules=list(pl_module.model.children())[-self.num_layers:],
            optimizer=optimizer,
            train_bn=self.train_bn,
        )


class VideoClassifier(ClassificationTask):
    """Task that classifies videos.

    Args:
        num_classes: Number of classes to classify.
        model: A string mapped to ``pytorch_video`` models or ``nn.Module``, defaults to ``"slowfast_r50"``.
        pretrained: Use a pretrained backbone, defaults to ``True``.
        loss_fn: Loss function for training, defaults to :func:`torch.nn.functional.cross_entropy`.
        optimizer: Optimizer to use for training, defaults to :class:`torch.optim.SGD`.
        metrics: Metrics to compute for training and evaluation,
            defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
    """

    models: FlashRegistry = _VIDEO_CLASSIFIER_MODELS

    def __init__(
        self,
        num_classes: int,
        model: Union[str, nn.Module] = "slow_r50",
        model_kwargs: Optional[Dict] = None,
        pretrained: bool = True,
        loss_fn: Callable = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        metrics: Union[Callable, Mapping, Sequence, None] = Accuracy(),
        learning_rate: float = 1e-3,
        head: Optional[Union[FunctionType, nn.Module]] = None,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

        if not model_kwargs:
            model_kwargs = {}

        model_kwargs["pretrained"] = pretrained
        model_kwargs["head_activation"] = None

        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, str):
            self.model = self.models.get(model)(**model_kwargs)
            num_features = self.model.blocks[-1].proj.out_features
        else:
            raise MisconfigurationException(f"model should be either a string or a nn.Module. Found: {model}")

        self.head = head or nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )

    def on_train_start(self) -> None:
        if self.trainer.accelerator_connector.is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.dataset
            encoded_dataset._video_sampler = DistributedSampler(encoded_dataset._labeled_videos)
        super().on_train_start()

    def on_train_epoch_start(self) -> None:
        if self.trainer.accelerator_connector.is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.dataset
            encoded_dataset._video_sampler.set_epoch(self.trainer.current_epoch)
        super().on_train_epoch_start()

    def step(self, batch: Any, batch_idx: int) -> Any:
        return super().step((batch["video"], batch["label"]), batch_idx)

    def forward(self, x: Any) -> Any:
        x = self.model(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        predictions = self(batch["video"])
        return predictions

    def configure_finetune_callback(self) -> List[Callback]:
        return [VideoClassifierFinetuning()]
