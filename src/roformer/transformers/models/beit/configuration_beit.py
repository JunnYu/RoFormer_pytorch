# coding=utf-8
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" BEiT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/beit-base-patch16-224-in22k": "https://huggingface.co/microsoft/beit-base-patch16-224-in22k/resolve/main/config.json",
    # See all BEiT models at https://huggingface.co/models?filter=beit
}


class BeitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BeitModel`. It is used to
    instantiate an BEiT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BEiT
    `microsoft/beit-base-patch16-224-in22k <https://huggingface.co/microsoft/beit-base-patch16-224-in22k>`__
    architecture.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 8092):
            Vocabulary size of the BEiT model. Defines the number of different image tokens that can be used during
            pre-training.
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
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to :obj:`16`):
            The size (resolution) of each patch.
        num_channels (:obj:`int`, `optional`, defaults to :obj:`3`):
            The number of input channels.
        use_mask_token (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use a mask token for masked image modeling.
        use_absolute_position_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use BERT-style absolute position embeddings.
        use_relative_position_bias (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use T5-style relative position embeddings in the self-attention layers.
        use_shared_relative_position_bias (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use the same relative position embeddings across all self-attention layers of the Transformer.
        layer_scale_init_value (:obj:`float`, `optional`, defaults to 0.1):
            Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.
        drop_path_rate (:obj:`float`, `optional`, defaults to 0.1):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_mean_pooling (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
            CLS token, before applying the classification head.
        out_indices (:obj:`List[int]`, `optional`, defaults to :obj:`[3, 5, 7, 11]`):
            Indices of the feature maps to use for semantic segmentation.
        pool_scales (:obj:`Tuple[int]`, `optional`, defaults to :obj:`[1, 2, 3, 6]`):
            Pooling scales used in Pooling Pyramid Module applied on the last feature map.
        use_auxiliary_head (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to use an auxiliary head during training.
        auxiliary_loss_weight (:obj:`float`, `optional`, defaults to 0.4):
            Weight of the cross-entropy loss of the auxiliary head.
        auxiliary_channels (:obj:`int`, `optional`, defaults to 256):
            Number of channels to use in the auxiliary head.
        auxiliary_num_convs (:obj:`int`, `optional`, defaults to 1):
            Number of convolutional layers to use in the auxiliary head.
        auxiliary_concat_input (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to concatenate the output of the auxiliary head with the input before the classification layer.
        semantic_loss_ignore_index (:obj:`int`, `optional`, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example::

        >>> from transformers import BeitModel, BeitConfig

        >>> # Initializing a BEiT beit-base-patch16-224-in22k style configuration
        >>> configuration = BeitConfig()

        >>> # Initializing a model from the beit-base-patch16-224-in22k style configuration
        >>> model = BeitModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "beit"

    def __init__(
        self,
        vocab_size=8192,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=224,
        patch_size=16,
        num_channels=3,
        use_mask_token=False,
        use_absolute_position_embeddings=False,
        use_relative_position_bias=False,
        use_shared_relative_position_bias=False,
        layer_scale_init_value=0.1,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        out_indices=[3, 5, 7, 11],
        pool_scales=[1, 2, 3, 6],
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        auxiliary_channels=256,
        auxiliary_num_convs=1,
        auxiliary_concat_input=False,
        semantic_loss_ignore_index=255,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_relative_position_bias = use_relative_position_bias
        self.use_shared_relative_position_bias = use_shared_relative_position_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.use_mean_pooling = use_mean_pooling
        # decode head attributes (semantic segmentation)
        self.out_indices = out_indices
        self.pool_scales = pool_scales
        # auxiliary head attributes (semantic segmentation)
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
