# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
""" VisionTextDualEncoder model configuration """

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..clip.configuration_clip import CLIPVisionConfig


logger = logging.get_logger(__name__)


class VisionTextDualEncoderConfig(PretrainedConfig):
    r"""
    :class:`~transformers.VisionTextDualEncoderConfig` is the configuration class to store the configuration of a
    :class:`~transformers.VisionTextDualEncoderModel`. It is used to instantiate
    :class:`~transformers.VisionTextDualEncoderModel` model according to the specified arguments, defining the text
    model and vision model configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        text_config_dict (:obj:`dict`):
            Dictionary of configuration options that defines text model config.
        vision_config_dict (:obj:`dict`):
            Dictionary of configuration options that defines vison model config.
        projection_dim (:obj:`int`, `optional`, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (:obj:`float`, `optional`, defaults to 2.6592):
            The inital value of the `logit_scale` paramter. Default is used as per the original CLIP implementation.
        kwargs (`optional`):
            Dictionary of keyword arguments.

    Examples::

        >>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

        >>> # Initializing a BERT and ViT configuration
        >>> config_vision = ViTConfig()
        >>> config_text = BertConfig()

        >>> config = VisionTextDualEncoderConfig.from_vision_text_configs(config_vision, config_text, projection_dim=512)

        >>> # Initializing a BERT and ViT model
        >>> model = VisionTextDualEncoderModel(config=config)

        >>> # Accessing the model configuration
        >>> config_vision  = model.config.vision_config
        >>> config_text = model.config.text_config

        >>> # Saving the model, including its configuration
        >>> model.save_pretrained('my-model')

        >>> # loading model and config from pretrained folder
        >>> vision_text_config = VisionTextDualEncoderConfig.from_pretrained('vit-bert')
        >>> model = VisionTextDualEncoderModel.from_pretrained('vit-bert', config=vision_text_config)
    """

    model_type = "vision-text-dual-encoder"
    is_composition = True

    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")

        vision_config = kwargs.pop("vision_config")
        text_config = kwargs.pop("text_config")

        vision_model_type = vision_config.pop("model_type")
        text_model_type = text_config.pop("model_type")

        if vision_model_type == "clip":
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config).vision_config
        elif vision_model_type == "clip_vision_model":
            self.vision_config = CLIPVisionConfig(**vision_config)
        else:
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config)

        self.text_config = AutoConfig.for_model(text_model_type, **text_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_vision_text_configs(cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a :class:`VisionTextDualEncoderConfig` (or a derived class) from text model configuration and
        vision model configuration.

        Returns:
            :class:`VisionTextDualEncoderConfig`: An instance of a configuration object
        """

        return cls(vision_config=vision_config.to_dict(), text_config=text_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
