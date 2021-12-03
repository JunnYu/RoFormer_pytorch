# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tf_available, is_torch_available


_import_structure = {
    "configuration_flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig"],
    "tokenization_flaubert": ["FlaubertTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_flaubert"] = [
        "FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlaubertForMultipleChoice",
        "FlaubertForQuestionAnswering",
        "FlaubertForQuestionAnsweringSimple",
        "FlaubertForSequenceClassification",
        "FlaubertForTokenClassification",
        "FlaubertModel",
        "FlaubertWithLMHeadModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_flaubert"] = [
        "TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFFlaubertForMultipleChoice",
        "TFFlaubertForQuestionAnsweringSimple",
        "TFFlaubertForSequenceClassification",
        "TFFlaubertForTokenClassification",
        "TFFlaubertModel",
        "TFFlaubertPreTrainedModel",
        "TFFlaubertWithLMHeadModel",
    ]


if TYPE_CHECKING:
    from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
    from .tokenization_flaubert import FlaubertTokenizer

    if is_torch_available():
        from .modeling_flaubert import (
            FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaubertForMultipleChoice,
            FlaubertForQuestionAnswering,
            FlaubertForQuestionAnsweringSimple,
            FlaubertForSequenceClassification,
            FlaubertForTokenClassification,
            FlaubertModel,
            FlaubertWithLMHeadModel,
        )

    if is_tf_available():
        from .modeling_tf_flaubert import (
            TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFFlaubertForMultipleChoice,
            TFFlaubertForQuestionAnsweringSimple,
            TFFlaubertForSequenceClassification,
            TFFlaubertForTokenClassification,
            TFFlaubertModel,
            TFFlaubertPreTrainedModel,
            TFFlaubertWithLMHeadModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
