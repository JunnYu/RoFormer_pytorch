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

from ...file_utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
    "tokenization_gpt2": ["GPT2Tokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]

if is_torch_available():
    _import_structure["modeling_gpt2"] = [
        "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPT2DoubleHeadsModel",
        "GPT2ForSequenceClassification",
        "GPT2ForTokenClassification",
        "GPT2LMHeadModel",
        "GPT2Model",
        "GPT2PreTrainedModel",
        "load_tf_weights_in_gpt2",
    ]

if is_tf_available():
    _import_structure["modeling_tf_gpt2"] = [
        "TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFGPT2DoubleHeadsModel",
        "TFGPT2ForSequenceClassification",
        "TFGPT2LMHeadModel",
        "TFGPT2MainLayer",
        "TFGPT2Model",
        "TFGPT2PreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"]

if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig
    from .tokenization_gpt2 import GPT2Tokenizer

    if is_tokenizers_available():
        from .tokenization_gpt2_fast import GPT2TokenizerFast

    if is_torch_available():
        from .modeling_gpt2 import (
            GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPT2DoubleHeadsModel,
            GPT2ForSequenceClassification,
            GPT2ForTokenClassification,
            GPT2LMHeadModel,
            GPT2Model,
            GPT2PreTrainedModel,
            load_tf_weights_in_gpt2,
        )

    if is_tf_available():
        from .modeling_tf_gpt2 import (
            TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGPT2DoubleHeadsModel,
            TFGPT2ForSequenceClassification,
            TFGPT2LMHeadModel,
            TFGPT2MainLayer,
            TFGPT2Model,
            TFGPT2PreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_gpt2 import FlaxGPT2LMHeadModel, FlaxGPT2Model, FlaxGPT2PreTrainedModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
