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
from contextlib import contextmanager
from inspect import signature
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING

from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.warning_utils import rank_zero_warn
from torch.utils.data import Dataset

from flash.data.process import Preprocess
from flash.data.utils import _STAGES_PREFIX, _STAGES_PREFIX_VALUES, CurrentRunningStageFuncContext

if TYPE_CHECKING:
    from flash.data.data_pipeline import DataPipeline


class AutoDataset(Dataset):

    DATASET_KEY = "dataset"
    """
        This class is used to encapsulate a Preprocess Object ``load_data`` and ``load_sample`` functions.
        ``load_data`` will be called within the ``__init__`` function of the AutoDataset if ``running_stage``
        is provided and ``load_sample`` within ``__getitem__`` function.
    """

    def __init__(
        self,
        data: Any,
        load_data: Optional[Callable] = None,
        load_sample: Optional[Callable] = None,
        data_pipeline: Optional['DataPipeline'] = None,
        running_stage: Optional[RunningStage] = None
    ) -> None:
        super().__init__()

        if load_data or load_sample:
            if data_pipeline:
                rank_zero_warn(
                    "``datapipeline`` is specified but load_sample and/or load_data are also specified. "
                    "Won't use datapipeline"
                )
        # initial states
        self._load_data_called = False
        self._running_stage = None

        self.data = data
        self.data_pipeline = data_pipeline
        self.load_data = load_data
        self.load_sample = load_sample

        # trigger the setup only if `running_stage` is provided
        self.running_stage = running_stage

    @property
    def running_stage(self) -> Optional[RunningStage]:
        return self._running_stage

    @running_stage.setter
    def running_stage(self, running_stage: RunningStage) -> None:
        if self._running_stage != running_stage or (not self._running_stage):
            self._running_stage = running_stage
            self._load_data_context = CurrentRunningStageFuncContext(self._running_stage, "load_data", self.preprocess)
            self._load_sample_context = CurrentRunningStageFuncContext(
                self._running_stage, "load_sample", self.preprocess
            )
            self._setup(running_stage)

    @property
    def preprocess(self) -> Optional[Preprocess]:
        if self.data_pipeline is not None:
            return self.data_pipeline._preprocess_pipeline

    def _call_load_data(self, data: Any) -> Iterable:
        parameters = signature(self.load_data).parameters
        if len(parameters) > 1 and self.DATASET_KEY in parameters:
            return self.load_data(data, self)
        else:
            return self.load_data(data)

    def _call_load_sample(self, sample: Any) -> Any:
        parameters = signature(self.load_sample).parameters
        if len(parameters) > 1 and self.DATASET_KEY in parameters:
            return self.load_sample(sample, self)
        else:
            return self.load_sample(sample)

    def _setup(self, stage: Optional[RunningStage]) -> None:
        assert not stage or _STAGES_PREFIX[stage] in _STAGES_PREFIX_VALUES
        previous_load_data = self.load_data.__code__ if self.load_data else None

        if self._running_stage and self.data_pipeline and (not self.load_data or not self.load_sample) and stage:
            self.load_data = getattr(
                self.preprocess,
                self.data_pipeline._resolve_function_hierarchy('load_data', self.preprocess, stage, Preprocess)
            )
            self.load_sample = getattr(
                self.preprocess,
                self.data_pipeline._resolve_function_hierarchy('load_sample', self.preprocess, stage, Preprocess)
            )
        if self.load_data and (previous_load_data != self.load_data.__code__ or not self._load_data_called):
            if previous_load_data:
                rank_zero_warn(
                    "The load_data function of the Autogenerated Dataset changed. "
                    "This is not expected! Preloading Data again to ensure compatibility. This may take some time."
                )
            with self._load_data_context:
                self.preprocessed_data = self._call_load_data(self.data)
            self._load_data_called = True

    def __getitem__(self, index: int) -> Any:
        if not self.load_sample and not self.load_data:
            raise RuntimeError("`__getitem__` for `load_sample` and `load_data` could not be inferred.")
        if self.load_sample:
            with self._load_sample_context:
                return self._call_load_sample(self.preprocessed_data[index])
        return self.preprocessed_data[index]

    def __len__(self) -> int:
        if not self.load_sample and not self.load_data:
            raise RuntimeError("`__len__` for `load_sample` and `load_data` could not be inferred.")
        return len(self.preprocessed_data)