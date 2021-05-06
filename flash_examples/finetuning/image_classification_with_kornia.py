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
import random

import kornia as K
import matplotlib.pyplot as plt
import numpy as np
# import our libraries
import torch
from PIL import Image

import flash
from flash.data.utils import download_data
from flash.vision import ImageClassificationData, ImageClassifier

print(K.__version__)

import pytorch_lightning as pl

print(pl.__version__)

import flash

print(flash.__version__)

download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
    batch_size=8,
)


def show_batch(dm: ImageClassificationData):
    # fetch data from the training data loader
    img, label = next(iter(dm.train_dataloader()))

    cols: int = 4
    rows: int = img.shape[0] // cols

    fig, axs = plt.subplots(rows, cols)
    fig.suptitle('Before forward')

    for i, ax in enumerate(axs.ravel()):
        _img, _label = img[i], label[i]
        ax.imshow(K.tensor_to_image(_img))
        ax.set_title(_label)
        ax.axis('off')
    plt.show()


#show_batch(datamodule)

import os
from typing import Dict, List

import pandas as pd

DATA_ROOT: str = datamodule.train_dataset.data

classes: List[str] = os.listdir(DATA_ROOT)

labels_dist: Dict[str, int] = {}

for class_name in classes:
    class_files: List[str] = os.listdir(os.path.join(DATA_ROOT, class_name))
    labels_dist[class_name] = [len(class_files)]

train_df = pd.DataFrame(labels_dist)
print(train_df.head())

# train_df.plot(kind="bar")

# augmentations

import kornia as K
import torch
import torch.nn as nn

# Define the augmentations pipeline

DATA_MEAN = [0.485, 0.456, 0.406]
DATA_STD = [0.229, 0.224, 0.225]


class Preprocess(nn.Module):
    """Applies the processing to the image in the worker before collate."""

    def __init__(self, image_size, mode: str):
        super().__init__()
        self.mode = mode
        self.resize = K.geometry.Resize(image_size)  # use this better to see whole image
        self.crop = K.augmentation.RandomResizedCrop(image_size)
        self.norm = K.augmentation.Normalize(
            torch.tensor(DATA_MEAN),
            torch.tensor(DATA_STD),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, x.shape
        if self.mode == 'train':
            #out = self.crop(x)
            out = self.resize(x)
        elif self.mode == 'test':
            out = self.resize(x)
        return self.norm(out)


class Augmentation(nn.Module):
    """Applies random augmentation to a batch of images."""

    def __init__(self, viz: bool = False):
        super().__init__()
        self.viz = viz
        '''self.geometric = [
            K.augmentation.RandomAffine(60., p=0.5),
            K.augmentation.RandomPerspective(0.4, p=0.5),
        ]'''
        self.augmentations = nn.Sequential(
            K.augmentation.RandomChannelShuffle(p=0.5),
            K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.4, p=0.5),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
        )
        self.denorm = K.augmentation.Denormalize(
            torch.tensor(DATA_MEAN),
            torch.tensor(DATA_STD),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4, x.shape
        #out = random.choice(self.geometric)(x)
        out = self.augmentations(x)
        if self.viz:
            out = self.denorm(out)
        return out


from typing import Callable

TRAIN_TRANSFORMS: Dict[str, nn.Module] = {
    "post_tensor_transform": Preprocess((196, 196), mode='train'),
    #"per_batch_transform": Augmentation(),
    # Use the one below when you want to train and perform the
    # the data augmentation on device (GPU/TPU).
    "per_batch_transform_on_device": Augmentation(),
}
VALID_TRANSFORMS: Dict[str, nn.Module] = {
    "post_tensor_transform": Preprocess((196, 196), mode='test'),
}

# 2. Load the data
datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/test/",
    batch_size=8,
    train_transform=TRAIN_TRANSFORMS,
    val_transform=VALID_TRANSFORMS,
)
#datamodule.show_train_batch([
#    'load_sample', 'per_batch_transform'
#])

# 3. Build the model
model = ImageClassifier(
    num_classes=datamodule.num_classes,
    backbone="dino_deits16",
    #backbone="resnet50",
    optimizer=torch.optim.SGD
    #optimizer=torch.optim.Adam
    #optimizer=torch.optim.AdamW
)

# 4. Create the trainer. Run once on data
trainer = flash.Trainer(max_epochs=100, gpus=1)

# 5. Finetune the model
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 6. Save it!
trainer.save_checkpoint("image_classification_model.pt")
