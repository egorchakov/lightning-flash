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
from typing import Optional, Union

from flash.data.process import Postprocess, Preprocess
from flash.text.seq2seq.core.data import Seq2SeqData


class TranslationData(Seq2SeqData):
    """Data module for Translation tasks."""

    @classmethod
    def from_files(
        cls,
        train_file,
        input: str = 'input',
        target: Optional[str] = None,
        filetype="csv",
        backbone="Helsinki-NLP/opus-mt-en-ro",
        use_fast: bool = True,
        val_file=None,
        test_file=None,
        predict_file=None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'max_length',
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
    ):
        """Creates a TranslateData object from files.

        Args:
            train_file: Path to training data.
            input: The field storing the source translation text.
            target: The field storing the target translation text.
            filetype: .csv or .json
            backbone: Tokenizer backbone to use, can use any HuggingFace tokenizer.
            val_file: Path to validation data.
            test_file: Path to test data.
            predict_file: Path to predict data.
            max_source_length: Maximum length of the source text. Any text longer will be truncated.
            max_target_length: Maximum length of the target text. Any text longer will be truncated.
            padding: Padding strategy for batches. Default is pad to maximum length.
            batch_size: The batchsize to use for parallel loading. Defaults to 8.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,
                or 0 for Darwin platform.

        Returns:
            TranslateData: The constructed data module.

        Examples::

            datamodule = TranslationData.from_files(
                train_file="data/wmt_en_ro/train.csv",
                val_file="data/wmt_en_ro/valid.csv",
                test_file="data/wmt_en_ro/test.csv",
                input="input",
                target="target",
                batch_size=1,
            )

        """
        return super().from_files(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            predict_file=predict_file,
            input=input,
            target=target,
            backbone=backbone,
            use_fast=use_fast,
            filetype=filetype,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            postprocess=postprocess,
        )

    @classmethod
    def from_file(
        cls,
        predict_file: str,
        input: str = 'input',
        target: Optional[str] = None,
        backbone="facebook/mbart-large-en-ro",
        filetype="csv",
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = 'longest',
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
    ):
        """Creates a TranslationData object from files.

        Args:
            predict_file: Path to prediction input file.
            input: The field storing the source translation text.
            target: The field storing the target translation text.
            backbone: Tokenizer backbone to use, can use any HuggingFace tokenizer.
            filetype: csv or json.
            max_source_length: Maximum length of the source text. Any text longer will be truncated.
            max_target_length: Maximum length of the target text. Any text longer will be truncated.
            padding: Padding strategy for batches. Default is pad to maximum length.
            batch_size: The batchsize to use for parallel loading. Defaults to 8.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads,


        Returns:
            Seq2SeqData: The constructed data module.

        """
        return super().from_file(
            predict_file=predict_file,
            input=input,
            target=target,
            backbone=backbone,
            filetype=filetype,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            postprocess=postprocess,
        )
