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
import contextlib
import os
import tempfile

import pytest
import torch
import torchvision.io as io
from torch.utils.data import SequentialSampler

import flash
from flash.utils.imports import _PYTORCHVIDEO_AVAILABLE

if _PYTORCHVIDEO_AVAILABLE:
    import kornia.augmentation as K
    from pytorchvideo.data.utils import thwc_to_cthw
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

    from flash.video import VideoClassificationData, VideoClassifier


def create_dummy_video_frames(num_frames: int, height: int, width: int):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc)**2 + (y - yc)**2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())
    return torch.stack(data, 0)


# https://github.com/facebookresearch/pytorchvideo/blob/4feccb607d7a16933d485495f91d067f177dd8db/tests/utils.py#L33
@contextlib.contextmanager
def temp_encoded_video(num_frames: int, fps: int, height=10, width=10, prefix=None):
    """
    Creates a temporary lossless, mp4 video with synthetic content. Uses a context which
    deletes the video after exit.
    """
    # Lossless options.
    video_codec = "libx264rgb"
    options = {"crf": "0"}
    data = create_dummy_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".mp4") as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, thwc_to_cthw(data).to(torch.float32)
    os.unlink(f.name)


@contextlib.contextmanager
def mock_encoded_video_dataset_file():
    """
    Creates a temporary mock encoded video dataset with 4 videos labeled from 0 - 4.
    Returns a labeled video file which points to this mock encoded video dataset, the
    ordered label and videos tuples and the video duration in seconds.
    """
    num_frames = 10
    fps = 5
    with temp_encoded_video(num_frames=num_frames, fps=fps) as (
        video_file_name_1,
        data_1,
    ):
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name_2,
            data_2,
        ):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(f"{video_file_name_1} 0\n".encode())
                f.write(f"{video_file_name_2} 1\n".encode())
                f.write(f"{video_file_name_1} 2\n".encode())
                f.write(f"{video_file_name_2} 3\n".encode())

            label_videos = [
                (0, data_1),
                (1, data_2),
                (2, data_1),
                (3, data_2),
            ]
            video_duration = num_frames / fps
            yield f.name, label_videos, video_duration


@pytest.mark.skipif(not _PYTORCHVIDEO_AVAILABLE, reason="PyTorchVideo isn't installed.")
def test_image_classifier_finetune(tmpdir):

    with mock_encoded_video_dataset_file() as (
        mock_csv,
        label_videos,
        total_duration,
    ):

        half_duration = total_duration / 2 - 1e-9

        datamodule = VideoClassificationData.from_paths(
            train_data_path=mock_csv,
            clip_sampler="uniform",
            clip_duration=half_duration,
            video_sampler=SequentialSampler,
            decode_audio=False,
        )

        for sample in datamodule.train_dataset.dataset:
            expected_t_shape = 5
            assert sample["video"].shape[1] == expected_t_shape

        assert len(VideoClassifier.available_models()) > 5

        train_transform = {
            "post_tensor_transform": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([
                        UniformTemporalSubsample(8),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(244),
                        RandomHorizontalFlip(p=0.5),
                    ]),
                ),
            ]),
            "per_batch_transform_on_device": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=K.VideoSequential(
                        K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225])),
                        K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
                        data_format="BCTHW",
                        same_on_frame=False
                    )
                ),
            ]),
        }

        datamodule = VideoClassificationData.from_paths(
            train_data_path=mock_csv,
            clip_sampler="uniform",
            clip_duration=half_duration,
            video_sampler=SequentialSampler,
            decode_audio=False,
            train_transform=train_transform
        )

        model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False)

        trainer = flash.Trainer(fast_dev_run=True)

        trainer.finetune(model, datamodule=datamodule)
