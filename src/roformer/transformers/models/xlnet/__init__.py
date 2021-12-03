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

from ...file_utils import (
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"],
}

if is_sentencepiece_available():
    _import_structure["tokenization_xlnet"] = ["XLNetTokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_xlnet_fast"] = ["XLNetTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_xlnet"] = [
        "XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLNetForMultipleChoice",
        "XLNetForQuestionAnswering",
        "XLNetForQuestionAnsweringSimple",
        "XLNetForSequenceClassification",
        "XLNetForTokenClassification",
        "XLNetLMHeadModel",
        "XLNetModel",
        "XLNetPreTrainedModel",
        "load_tf_weights_in_xlnet",
    ]

if is_tf_available():
    _import_structure["modeling_tf_xlnet"] = [
        "TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXLNetForMultipleChoice",
        "TFXLNetForQuestionAnsweringSimple",
        "TFXLNetForSequenceClassification",
        "TFXLNetForTokenClassification",
        "TFXLNetLMHeadModel",
        "TFXLNetMainLayer",
        "TFXLNetModel",
        "TFXLNetPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig

    if is_sentencepiece_available():
        from .tokenization_xlnet import XLNetTokenizer

    if is_tokenizers_available():
        from .tokenization_xlnet_fast import XLNetTokenizerFast

    if is_torch_available():
        from .modeling_xlnet import (
            XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLNetForMultipleChoice,
            XLNetForQuestionAnswering,
            XLNetForQuestionAnsweringSimple,
            XLNetForSequenceClassification,
            XLNetForTokenClassification,
            XLNetLMHeadModel,
            XLNetModel,
            XLNetPreTrainedModel,
            load_tf_weights_in_xlnet,
        )

    if is_tf_available():
        from .modeling_tf_xlnet import (
            TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLNetForMultipleChoice,
            TFXLNetForQuestionAnsweringSimple,
            TFXLNetForSequenceClassification,
            TFXLNetForTokenClassification,
            TFXLNetLMHeadModel,
            TFXLNetMainLayer,
            TFXLNetModel,
            TFXLNetPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
