# coding=utf-8
# Copyright The HuggingFace team. All rights reserved.
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
""" ConvBERT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/config.json",
    "YituTech/conv-bert-medium-small": "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/config.json",
    "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/config.json",
    # See all ConvBERT models at https://huggingface.co/models?filter=convbert
}


class ConvBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ConvBertModel`. It is used to
    instantiate an ConvBERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the ConvBERT `conv-bert-base
    <https://huggingface.co/YituTech/conv-bert-base>`__ architecture. Configuration objects inherit from
    :class:`~transformers.PretrainedConfig` and can be used to control the model outputs. Read the documentation from
    :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ConvBERT model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.ConvBertModel` or
            :class:`~transformers.TFConvBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.ConvBertModel`
            or :class:`~transformers.TFConvBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        head_ratio (:obj:`int`, `optional`, defaults to 2):
            Ratio gamma to reduce the number of attention heads.
        num_groups (:obj:`int`, `optional`, defaults to 1):
            The number of groups for grouped linear layers for ConvBert model
        conv_kernel_size (:obj:`int`, `optional`, defaults to 9):
            The size of the convolutional kernel.
        classifier_dropout (:obj:`float`, `optional`):
            The dropout ratio for the classification head.

    Example::
        >>> from transformers import ConvBertModel, ConvBertConfig
        >>> # Initializing a ConvBERT convbert-base-uncased style configuration
        >>> configuration = ConvBertConfig()
        >>> # Initializing a model from the convbert-base-uncased style configuration
        >>> model = ConvBertModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "convbert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        is_encoder_decoder=False,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        embedding_size=768,
        head_ratio=2,
        conv_kernel_size=9,
        num_groups=1,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.head_ratio = head_ratio
        self.conv_kernel_size = conv_kernel_size
        self.num_groups = num_groups
        self.classifier_dropout = classifier_dropout
