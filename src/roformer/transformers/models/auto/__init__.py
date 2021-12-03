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

from ...file_utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "auto_factory": ["get_values"],
    "configuration_auto": ["ALL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "AutoConfig"],
    "feature_extraction_auto": ["FEATURE_EXTRACTOR_MAPPING", "AutoFeatureExtractor"],
    "processing_auto": ["PROCESSOR_MAPPING", "AutoProcessor"],
    "tokenization_auto": ["TOKENIZER_MAPPING", "AutoTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_auto"] = [
        "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
        "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING",
        "MODEL_FOR_CAUSAL_LM_MAPPING",
        "MODEL_FOR_CTC_MAPPING",
        "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
        "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
        "MODEL_FOR_MASKED_LM_MAPPING",
        "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "MODEL_FOR_OBJECT_DETECTION_MAPPING",
        "MODEL_FOR_PRETRAINING_MAPPING",
        "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
        "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "MODEL_FOR_VISION_2_SEQ_MAPPING",
        "MODEL_MAPPING",
        "MODEL_WITH_LM_HEAD_MAPPING",
        "AutoModel",
        "AutoModelForAudioClassification",
        "AutoModelForCausalLM",
        "AutoModelForCTC",
        "AutoModelForImageClassification",
        "AutoModelForImageSegmentation",
        "AutoModelForMaskedLM",
        "AutoModelForMultipleChoice",
        "AutoModelForNextSentencePrediction",
        "AutoModelForObjectDetection",
        "AutoModelForPreTraining",
        "AutoModelForQuestionAnswering",
        "AutoModelForSeq2SeqLM",
        "AutoModelForSequenceClassification",
        "AutoModelForSpeechSeq2Seq",
        "AutoModelForTableQuestionAnswering",
        "AutoModelForTokenClassification",
        "AutoModelForVision2Seq",
        "AutoModelWithLMHead",
    ]

if is_tf_available():
    _import_structure["modeling_tf_auto"] = [
        "TF_MODEL_FOR_CAUSAL_LM_MAPPING",
        "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
        "TF_MODEL_FOR_MASKED_LM_MAPPING",
        "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "TF_MODEL_FOR_PRETRAINING_MAPPING",
        "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
        "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "TF_MODEL_MAPPING",
        "TF_MODEL_WITH_LM_HEAD_MAPPING",
        "TFAutoModel",
        "TFAutoModelForCausalLM",
        "TFAutoModelForImageClassification",
        "TFAutoModelForMaskedLM",
        "TFAutoModelForMultipleChoice",
        "TFAutoModelForPreTraining",
        "TFAutoModelForQuestionAnswering",
        "TFAutoModelForSeq2SeqLM",
        "TFAutoModelForSequenceClassification",
        "TFAutoModelForTableQuestionAnswering",
        "TFAutoModelForTokenClassification",
        "TFAutoModelWithLMHead",
    ]

if is_flax_available():
    _import_structure["modeling_flax_auto"] = [
        "FLAX_MODEL_FOR_CAUSAL_LM_MAPPING",
        "FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_MASKED_LM_MAPPING",
        "FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "FLAX_MODEL_FOR_PRETRAINING_MAPPING",
        "FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING",
        "FLAX_MODEL_MAPPING",
        "FlaxAutoModel",
        "FlaxAutoModelForCausalLM",
        "FlaxAutoModelForImageClassification",
        "FlaxAutoModelForMaskedLM",
        "FlaxAutoModelForMultipleChoice",
        "FlaxAutoModelForNextSentencePrediction",
        "FlaxAutoModelForPreTraining",
        "FlaxAutoModelForQuestionAnswering",
        "FlaxAutoModelForSeq2SeqLM",
        "FlaxAutoModelForSequenceClassification",
        "FlaxAutoModelForTokenClassification",
        "FlaxAutoModelForVision2Seq",
    ]


if TYPE_CHECKING:
    from .auto_factory import get_values
    from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
    from .feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
    from .processing_auto import PROCESSOR_MAPPING, AutoProcessor
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer

    if is_torch_available():
        from .modeling_auto import (
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_CTC_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_OBJECT_DETECTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_FOR_VISION_2_SEQ_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoModel,
            AutoModelForAudioClassification,
            AutoModelForCausalLM,
            AutoModelForCTC,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForMaskedLM,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForObjectDetection,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForSpeechSeq2Seq,
            AutoModelForTableQuestionAnswering,
            AutoModelForTokenClassification,
            AutoModelForVision2Seq,
            AutoModelWithLMHead,
        )

    if is_tf_available():
        from .modeling_tf_auto import (
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TFAutoModel,
            TFAutoModelForCausalLM,
            TFAutoModelForImageClassification,
            TFAutoModelForMaskedLM,
            TFAutoModelForMultipleChoice,
            TFAutoModelForPreTraining,
            TFAutoModelForQuestionAnswering,
            TFAutoModelForSeq2SeqLM,
            TFAutoModelForSequenceClassification,
            TFAutoModelForTableQuestionAnswering,
            TFAutoModelForTokenClassification,
            TFAutoModelWithLMHead,
        )

    if is_flax_available():
        from .modeling_flax_auto import (
            FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_MASKED_LM_MAPPING,
            FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            FLAX_MODEL_FOR_PRETRAINING_MAPPING,
            FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
            FLAX_MODEL_MAPPING,
            FlaxAutoModel,
            FlaxAutoModelForCausalLM,
            FlaxAutoModelForImageClassification,
            FlaxAutoModelForMaskedLM,
            FlaxAutoModelForMultipleChoice,
            FlaxAutoModelForNextSentencePrediction,
            FlaxAutoModelForPreTraining,
            FlaxAutoModelForQuestionAnswering,
            FlaxAutoModelForSeq2SeqLM,
            FlaxAutoModelForSequenceClassification,
            FlaxAutoModelForTokenClassification,
            FlaxAutoModelForVision2Seq,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
