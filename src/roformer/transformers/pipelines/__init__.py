# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

import io
import json

# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import http_get, is_tf_available, is_torch_available
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import logging
from .audio_classification import AudioClassificationPipeline
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from .base import (
    ArgumentHandler,
    CsvPipelineDataFormat,
    JsonPipelineDataFormat,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    PipelineException,
    get_default_model,
    infer_framework_load_model,
)
from .conversational import Conversation, ConversationalPipeline
from .feature_extraction import FeatureExtractionPipeline
from .fill_mask import FillMaskPipeline
from .image_classification import ImageClassificationPipeline
from .image_segmentation import ImageSegmentationPipeline
from .object_detection import ObjectDetectionPipeline
from .question_answering import QuestionAnsweringArgumentHandler, QuestionAnsweringPipeline
from .table_question_answering import TableQuestionAnsweringArgumentHandler, TableQuestionAnsweringPipeline
from .text2text_generation import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .token_classification import (
    AggregationStrategy,
    NerPipeline,
    TokenClassificationArgumentHandler,
    TokenClassificationPipeline,
)
from .zero_shot_classification import ZeroShotClassificationArgumentHandler, ZeroShotClassificationPipeline


if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTableQuestionAnswering,
        TFAutoModelForTokenClassification,
    )

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import (
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        AutoModel,
        AutoModelForAudioClassification,
        AutoModelForCausalLM,
        AutoModelForCTC,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForMaskedLM,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification,
    )
if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


# Register all the supported tasks here
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
}
SUPPORTED_TASKS = {
    "audio-classification": {
        "impl": AudioClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForAudioClassification,) if is_torch_available() else (),
        "default": {"model": {"pt": "superb/wav2vec2-base-superb-ks"}},
    },
    "automatic-speech-recognition": {
        "impl": AutomaticSpeechRecognitionPipeline,
        "tf": (),
        "pt": (AutoModelForCTC, AutoModelForSpeechSeq2Seq) if is_torch_available() else (),
        "default": {"model": {"pt": "facebook/wav2vec2-base-960h"}},
    },
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": (TFAutoModel,) if is_tf_available() else (),
        "pt": (AutoModel,) if is_torch_available() else (),
        "default": {"model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"}},
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
                "tf": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "tf": (TFAutoModelForTokenClassification,) if is_tf_available() else (),
        "pt": (AutoModelForTokenClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
        },
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": (TFAutoModelForQuestionAnswering,) if is_tf_available() else (),
        "pt": (AutoModelForQuestionAnswering,) if is_torch_available() else (),
        "default": {
            "model": {"pt": "distilbert-base-cased-distilled-squad", "tf": "distilbert-base-cased-distilled-squad"},
        },
    },
    "table-question-answering": {
        "impl": TableQuestionAnsweringPipeline,
        "pt": (AutoModelForTableQuestionAnswering,) if is_torch_available() else (),
        "tf": (TFAutoModelForTableQuestionAnswering,) if is_tf_available() else (),
        "default": {
            "model": {
                "pt": "google/tapas-base-finetuned-wtq",
                "tokenizer": "google/tapas-base-finetuned-wtq",
                "tf": "google/tapas-base-finetuned-wtq",
            },
        },
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "tf": (TFAutoModelForMaskedLM,) if is_tf_available() else (),
        "pt": (AutoModelForMaskedLM,) if is_torch_available() else (),
        "default": {"model": {"pt": "distilroberta-base", "tf": "distilroberta-base"}},
    },
    "summarization": {
        "impl": SummarizationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {"model": {"pt": "sshleifer/distilbart-cnn-12-6", "tf": "t5-small"}},
    },
    # This task is a special case as it's parametrized by SRC, TGT languages.
    "translation": {
        "impl": TranslationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {
            ("en", "fr"): {"model": {"pt": "t5-base", "tf": "t5-base"}},
            ("en", "de"): {"model": {"pt": "t5-base", "tf": "t5-base"}},
            ("en", "ro"): {"model": {"pt": "t5-base", "tf": "t5-base"}},
        },
    },
    "text2text-generation": {
        "impl": Text2TextGenerationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {"model": {"pt": "t5-base", "tf": "t5-base"}},
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "tf": (TFAutoModelForCausalLM,) if is_tf_available() else (),
        "pt": (AutoModelForCausalLM,) if is_torch_available() else (),
        "default": {"model": {"pt": "gpt2", "tf": "gpt2"}},
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
            "config": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
            "tokenizer": {"pt": "facebook/bart-large-mnli", "tf": "roberta-large-mnli"},
        },
    },
    "conversational": {
        "impl": ConversationalPipeline,
        "tf": (TFAutoModelForSeq2SeqLM, TFAutoModelForCausalLM) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM, AutoModelForCausalLM) if is_torch_available() else (),
        "default": {"model": {"pt": "microsoft/DialoGPT-medium", "tf": "microsoft/DialoGPT-medium"}},
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForImageClassification,) if is_torch_available() else (),
        "default": {"model": {"pt": "google/vit-base-patch16-224"}},
    },
    "image-segmentation": {
        "impl": ImageSegmentationPipeline,
        "tf": (),
        "pt": (AutoModelForImageSegmentation,) if is_torch_available() else (),
        "default": {"model": {"pt": "facebook/detr-resnet-50-panoptic"}},
    },
    "object-detection": {
        "impl": ObjectDetectionPipeline,
        "tf": (),
        "pt": (AutoModelForObjectDetection,) if is_torch_available() else (),
        "default": {"model": {"pt": "facebook/detr-resnet-50"}},
    },
}


def get_task(model: str, use_auth_token: Optional[str] = None) -> str:
    tmp = io.BytesIO()
    headers = {}
    if use_auth_token:
        headers["Authorization"] = f"Bearer {use_auth_token}"

    try:
        http_get(f"https://huggingface.co/api/models/{model}", tmp, headers=headers)
        tmp.seek(0)
        body = tmp.read()
        data = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}")
    if "pipeline_tag" not in data:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    if data.get("library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {data['library_name']} not with transformers")
    task = data["pipeline_tag"]
    return task


def check_task(task: str) -> Tuple[Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"audio-classification"`
            - :obj:`"automatic-speech-recognition"`
            - :obj:`"conversational"`
            - :obj:`"feature-extraction"`
            - :obj:`"fill-mask"`
            - :obj:`"image-classification"`
            - :obj:`"question-answering"`
            - :obj:`"table-question-answering"`
            - :obj:`"text2text-generation"`
            - :obj:`"text-classification"` (alias :obj:`"sentiment-analysis" available)
            - :obj:`"text-generation"`
            - :obj:`"token-classification"` (alias :obj:`"ner"` available)
            - :obj:`"translation"`
            - :obj:`"translation_xx_to_yy"`
            - :obj:`"summarization"`
            - :obj:`"zero-shot-classification"`

    Returns:
        (task_defaults:obj:`dict`, task_options: (:obj:`tuple`, None)) The actual dictionary required to initialize the
        pipeline and some extra task options for parametrized tasks like "translation_XX_to_YY"


    """
    if task in TASK_ALIASES:
        task = TASK_ALIASES[task]
    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        return targeted_task, None

    if task.startswith("translation"):
        tokens = task.split("_")
        if len(tokens) == 4 and tokens[0] == "translation" and tokens[2] == "to":
            targeted_task = SUPPORTED_TASKS["translation"]
            return targeted_task, (tokens[1], tokens[3])
        raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")

    raise KeyError(
        f"Unknown task {task}, available tasks are {list(SUPPORTED_TASKS.keys()) + ['translation_XX_to_YY']}"
    )


def pipeline(
    task: str = None,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a :class:`~transformers.Pipeline`.

    Pipelines are made of:

        - A :doc:`tokenizer <tokenizer>` in charge of mapping raw textual input to token.
        - A :doc:`model <model>` to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"audio-classification"`: will return a :class:`~transformers.AudioClassificationPipeline`:.
            - :obj:`"automatic-speech-recognition"`: will return a
              :class:`~transformers.AutomaticSpeechRecognitionPipeline`:.
            - :obj:`"conversational"`: will return a :class:`~transformers.ConversationalPipeline`:.
            - :obj:`"feature-extraction"`: will return a :class:`~transformers.FeatureExtractionPipeline`:.
            - :obj:`"fill-mask"`: will return a :class:`~transformers.FillMaskPipeline`:.
            - :obj:`"image-classification"`: will return a :class:`~transformers.ImageClassificationPipeline`:.
            - :obj:`"question-answering"`: will return a :class:`~transformers.QuestionAnsweringPipeline`:.
            - :obj:`"table-question-answering"`: will return a :class:`~transformers.TableQuestionAnsweringPipeline`:.
            - :obj:`"text2text-generation"`: will return a :class:`~transformers.Text2TextGenerationPipeline`:.
            - :obj:`"text-classification"` (alias :obj:`"sentiment-analysis" available): will return a
              :class:`~transformers.TextClassificationPipeline`:.
            - :obj:`"text-generation"`: will return a :class:`~transformers.TextGenerationPipeline`:.
            - :obj:`"token-classification"` (alias :obj:`"ner"` available): will return a
              :class:`~transformers.TokenClassificationPipeline`:.
            - :obj:`"translation"`: will return a :class:`~transformers.TranslationPipeline`:.
            - :obj:`"translation_xx_to_yy"`: will return a :class:`~transformers.TranslationPipeline`:.
            - :obj:`"summarization"`: will return a :class:`~transformers.SummarizationPipeline`:.
            - :obj:`"zero-shot-classification"`: will return a :class:`~transformers.ZeroShotClassificationPipeline`:.

        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from :class:`~transformers.PreTrainedModel` (for PyTorch)
            or :class:`~transformers.TFPreTrainedModel` (for TensorFlow).

            If not provided, the default for the :obj:`task` will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If not provided, the default configuration file for the requested model will be used. That means that if
            :obj:`model` is given, its default configuration will be used. However, if :obj:`model` is not supplied,
            this :obj:`task`'s default model's config is used instead.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default tokenizer for the given :obj:`model` will be loaded (if it is a string). If
            :obj:`model` is not specified or not a string, then the default tokenizer for :obj:`config` is loaded (if
            it is a string). However, if :obj:`config` is also not given or not a string, then the default tokenizer
            for the given :obj:`task` will be loaded.
        feature_extractor (:obj:`str` or :obj:`~transformers.PreTrainedFeatureExtractor`, `optional`):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from
            :class:`~transformers.PreTrainedFeatureExtractor`.

            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.

            If not provided, the default feature extractor for the given :obj:`model` will be loaded (if it is a
            string). If :obj:`model` is not specified or not a string, then the default feature extractor for
            :obj:`config` is loaded (if it is a string). However, if :obj:`config` is also not given or not a string,
            then the default feature extractor for the given :obj:`task` will be loaded.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any identifier allowed by git.
        use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use a Fast tokenizer if possible (a :class:`~transformers.PreTrainedTokenizerFast`).
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's :obj:`from_pretrained(...,
            **model_kwargs)` function.
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

        >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        >>> # Sentiment analysis pipeline
        >>> pipeline('sentiment-analysis')

        >>> # Question answering pipeline, specifying the checkpoint identifier
        >>> pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
        >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> pipeline('ner', model=model, tokenizer=tokenizer)
    """
    if model_kwargs is None:
        model_kwargs = {}

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model"
            "being specified."
            "Please provide a task class or a model"
        )

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model "
            "as the provided tokenizer may not be compatible with the default model. "
            "Please provide a PreTrainedModel class or a path/identifier to a pretrained model when providing tokenizer."
        )
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model "
            "as the provided feature_extractor may not be compatible with the default model. "
            "Please provide a PreTrainedModel class or a path/identifier to a pretrained model when providing feature_extractor."
        )

    if task is None and model is not None:
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`."
                f"{model} is not a valid model_id."
            )
        task = get_task(model, use_auth_token)

    # Retrieve the task
    targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        # At that point framework might still be undetermined
        model = get_default_model(targeted_task, framework, task_options)
        logger.warning(f"No model was supplied, defaulted to {model} (https://huggingface.co/{model})")

    # Retrieve use_auth_token and add it to model_kwargs to be used in .from_pretrained
    model_kwargs["use_auth_token"] = model_kwargs.get("use_auth_token", use_auth_token)

    # Config is the primordial information item.
    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config, revision=revision, _from_pipeline=task, **model_kwargs)
    elif config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(model, revision=revision, _from_pipeline=task, **model_kwargs)

    model_name = model if isinstance(model, str) else None

    # Infer the framework from the model
    # Forced if framework already defined, inferred if it's None
    # Will load the correct model if possible
    model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]}
    framework, model = infer_framework_load_model(
        model,
        model_classes=model_classes,
        config=config,
        framework=framework,
        revision=revision,
        task=task,
        **model_kwargs,
    )

    model_config = model.config

    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
    load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None

    if task in {"audio-classification"}:
        # Audio classification will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False

    if load_tokenizer:
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer is None:
            if isinstance(model_name, str):
                tokenizer = model_name
            elif isinstance(config, str):
                tokenizer = config
            else:
                # Impossible to guess what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # Instantiate tokenizer if needed
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, revision=revision, use_fast=use_fast, _from_pipeline=task, **tokenizer_kwargs
            )

    if load_feature_extractor:
        # Try to infer feature extractor from model or config name (if provided as str)
        if feature_extractor is None:
            if isinstance(model_name, str):
                feature_extractor = model_name
            elif isinstance(config, str):
                feature_extractor = config
            else:
                # Impossible to guess what is the right feature_extractor here
                raise Exception(
                    "Impossible to guess which feature extractor to use. "
                    "Please provide a PreTrainedFeatureExtractor class or a path/identifier "
                    "to a pretrained feature extractor."
                )

        # Instantiate feature_extractor if needed
        if isinstance(feature_extractor, (str, tuple)):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                feature_extractor, revision=revision, _from_pipeline=task, **model_kwargs
            )

    if task == "translation" and model.config.task_specific_params:
        for key in model.config.task_specific_params:
            if key.startswith("translation"):
                task = key
                warnings.warn(
                    f'"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{task}"',
                    UserWarning,
                )
                break

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
