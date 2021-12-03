# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" CLIP model configuration """

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
    # See all CLIP models at https://huggingface.co/models?filter=clip
}


class CLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.CLIPModel`. It is used to
    instantiate an CLIP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLIP
    `openai/clip-vit-base-patch32 <https://huggingface.co/openai/clip-vit-base-patch32>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 49408):
            Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.CLIPModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (:obj:`int`, `optional`, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` :obj:`"quick_gelu"` are supported.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example::

        >>> from transformers import CLIPTextModel, CLIPTextConfig

        >>> # Initializing a CLIPTextModel with openai/clip-vit-base-patch32 style configuration
        >>> configuration = CLIPTextConfig()

        >>> # Initializing a CLIPTextConfig from the openai/clip-vit-base-patch32 style configuration
        >>> model = CLIPTextModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "clip_text_model"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


class CLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.CLIPModel`. It is used to
    instantiate an CLIP model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLIP
    `openai/clip-vit-base-patch32 <https://huggingface.co/openai/clip-vit-base-patch32>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (:obj:`int`, `optional`, defaults to 224):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` :obj:`"quick_gelu"` are supported.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example::

        >>> from transformers import CLIPVisionModel, CLIPVisionConfig

        >>> # Initializing a CLIPVisionModel with openai/clip-vit-base-patch32 style configuration
        >>> configuration = CLIPVisionConfig()

        >>> # Initializing a CLIPVisionModel model from the openai/clip-vit-base-patch32 style configuration
        >>> model = CLIPVisionModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class CLIPConfig(PretrainedConfig):
    r"""
    :class:`~transformers.CLIPConfig` is the configuration class to store the configuration of a
    :class:`~transformers.CLIPModel`. It is used to instantiate CLIP model according to the specified arguments,
    defining the text model and vision model configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        text_config_dict (:obj:`dict`, `optional`):
            Dictionary of configuration options used to initialize :class:`~transformers.CLIPTextConfig`.
        vision_config_dict (:obj:`dict`, `optional`):
            Dictionary of configuration options used to initialize :class:`~transformers.CLIPVisionConfig`.
        projection_dim (:obj:`int`, `optional`, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (:obj:`float`, `optional`, defaults to 2.6592):
            The inital value of the `logit_scale` paramter. Default is used as per the original CLIP implementation.
        kwargs (`optional`):
            Dictionary of keyword arguments.
    """

    model_type = "clip"
    is_composition = True

    def __init__(
        self,
        text_config_dict=None,
        vision_config_dict=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs
    ):
        super().__init__(text_config_dict=text_config_dict, vision_config_dict=vision_config_dict, **kwargs)

        if text_config_dict is None:
            text_config_dict = {}
            logger.info("text_config_dict is None. Initializing the CLIPTextConfig with default values.")

        if vision_config_dict is None:
            vision_config_dict = {}
            logger.info("vision_config_dict is None. initializing the CLIPVisionConfig with default values.")

        self.text_config = CLIPTextConfig(**text_config_dict)
        self.vision_config = CLIPVisionConfig(**vision_config_dict)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, **kwargs):
        r"""
        Instantiate a :class:`~transformers.CLIPConfig` (or a derived class) from clip text model configuration and
        clip vision model configuration.

        Returns:
            :class:`CLIPConfig`: An instance of a configuration object
        """

        return cls(text_config_dict=text_config.to_dict(), vision_config_dict=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
