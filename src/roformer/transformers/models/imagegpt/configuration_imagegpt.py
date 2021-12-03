# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
""" OpenAI ImageGPT configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/imagegpt-small": "",
    "openai/imagegpt-medium": "",
    "openai/imagegpt-large": "",
}


class ImageGPTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.ImageGPTModel` or a
    :class:`~transformers.TFImageGPTModel`. It is used to instantiate a GPT-2 model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the ImageGPT `small <https://huggingface.co/imagegpt>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 512):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.ImageGPTModel` or
            :class:`~transformers.TFImageGPTModel`.
        n_positions (:obj:`int`, `optional`, defaults to 32*32):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the embeddings and hidden states.
        n_layer (:obj:`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (:obj:`int`, `optional`, defaults to None):
            Dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd
        activation_function (:obj:`str`, `optional`, defaults to :obj:`"quick_gelu"`):
            Activation function (can be one of the activation functions defined in src/transformers/activations.py).
            Defaults to "quick_gelu".
        resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to additionally scale attention weights by ``1 / layer_idx + 1``.
        reorder_and_upcast_attn (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example::

        >>> from transformers import ImageGPTModel, ImageGPTConfig

        >>> # Initializing a ImageGPT configuration
        >>> configuration = ImageGPTConfig()

        >>> # Initializing a model from the configuration
        >>> model = ImageGPTModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "imagegpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=512 + 1,  # add one for start of sentence (sos) token
        n_positions=32 * 32,
        n_embd=512,
        n_layer=24,
        n_head=8,
        n_inner=None,
        activation_function="quick_gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        tie_word_embeddings=False,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
