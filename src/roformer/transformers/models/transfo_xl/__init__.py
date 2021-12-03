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
    "configuration_transfo_xl": ["TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP", "TransfoXLConfig"],
    "tokenization_transfo_xl": ["TransfoXLCorpus", "TransfoXLTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_transfo_xl"] = [
        "TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AdaptiveEmbedding",
        "TransfoXLForSequenceClassification",
        "TransfoXLLMHeadModel",
        "TransfoXLModel",
        "TransfoXLPreTrainedModel",
        "load_tf_weights_in_transfo_xl",
    ]

if is_tf_available():
    _import_structure["modeling_tf_transfo_xl"] = [
        "TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFAdaptiveEmbedding",
        "TFTransfoXLForSequenceClassification",
        "TFTransfoXLLMHeadModel",
        "TFTransfoXLMainLayer",
        "TFTransfoXLModel",
        "TFTransfoXLPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
    from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer

    if is_torch_available():
        from .modeling_transfo_xl import (
            TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            AdaptiveEmbedding,
            TransfoXLForSequenceClassification,
            TransfoXLLMHeadModel,
            TransfoXLModel,
            TransfoXLPreTrainedModel,
            load_tf_weights_in_transfo_xl,
        )

    if is_tf_available():
        from .modeling_tf_transfo_xl import (
            TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFAdaptiveEmbedding,
            TFTransfoXLForSequenceClassification,
            TFTransfoXLLMHeadModel,
            TFTransfoXLMainLayer,
            TFTransfoXLModel,
            TFTransfoXLPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
