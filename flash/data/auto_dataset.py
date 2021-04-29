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
from inspect import signature
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, Sequence, TYPE_CHECKING, TypeVar

import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.warning_utils import rank_zero_warn
from torch.utils.data import Dataset, IterableDataset

from flash.data.callback import ControlFlow
from flash.data.process import Preprocess
from flash.data.utils import _STAGES_PREFIX, _STAGES_PREFIX_VALUES, CurrentRunningStageFuncContext

if TYPE_CHECKING:
    from flash.data.data_pipeline import DataPipeline
    from flash.data.data_source import DataSource

DATA_TYPE = TypeVar('DATA_TYPE')


class BaseAutoDataset(Generic[DATA_TYPE]):

    DATASET_KEY = "dataset"
    """
        This class is used to encapsulate a Preprocess Object ``load_data`` and ``load_sample`` functions.
        ``load_data`` will be called within the ``__init__`` function of the AutoDataset if ``running_stage``
        is provided and ``load_sample`` within ``__getitem__``.
    """

    def __init__(
        self,
        data: DATA_TYPE,
        data_source: 'DataSource',
        running_stage: RunningStage,
        data_pipeline: Optional['DataPipeline'] = None,
    ) -> None:
        super().__init__()

        self.data = data
        self.data_source = data_source
        self.data_pipeline = data_pipeline

        self._running_stage = None
        self.running_stage = running_stage

    @property
    def running_stage(self) -> RunningStage:
        return self._running_stage

    @running_stage.setter
    def running_stage(self, running_stage: RunningStage) -> None:
        from flash.data.data_source import DataSource  # Hack to avoid circular import TODO: something better than this

        self._running_stage = running_stage

        self._load_sample_context = CurrentRunningStageFuncContext(self.running_stage, "load_sample", self.data_source)

        self.load_sample = getattr(
            self.data_source,
            self.data_pipeline._resolve_function_hierarchy(
                'load_sample',
                self.data_source,
                self.running_stage,
                DataSource,
            )
        )

    @property
    def preprocess(self) -> Optional[Preprocess]:
        if self.data_pipeline is not None:
            return self.data_pipeline._preprocess_pipeline

    @property
    def control_flow_callback(self) -> Optional[ControlFlow]:
        preprocess = self.preprocess
        if preprocess is not None:
            return ControlFlow(preprocess.callbacks)

    def _call_load_sample(self, sample: Any) -> Any:
        if self.load_sample:
            with self._load_sample_context:
                parameters = signature(self.load_sample).parameters
                if len(parameters) > 1 and self.DATASET_KEY in parameters:
                    sample = self.load_sample(sample, self)
                else:
                    sample = self.load_sample(sample)
                if self.control_flow_callback:
                    self.control_flow_callback.on_load_sample(sample, self.running_stage)
        return sample


class AutoDataset(BaseAutoDataset[Sequence[Any]], Dataset):

    def __getitem__(self, index: int) -> Any:
        return self._call_load_sample(self.data[index])

    def __len__(self) -> int:
        return len(self.data)


class IterableAutoDataset(BaseAutoDataset[Iterable[Any]], IterableDataset):

    def __iter__(self):
        self.data_iter = iter(self.data)
        return self

    def __next__(self) -> Any:
        return self._call_load_sample(next(self.data_iter))
