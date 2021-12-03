# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Allegro.pl, Facebook Inc. and the HuggingFace Inc. team.
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

from ...utils import logging
from ..bert.tokenization_bert import BasicTokenizer
from ..xlm.tokenization_xlm import XLMTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/vocab.json"
    },
    "merges_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/merges.txt"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"allegro/herbert-base-cased": 514}
PRETRAINED_INIT_CONFIGURATION = {}


class HerbertTokenizer(XLMTokenizer):
    """
    Construct a BPE tokenizer for HerBERT.

    Peculiarities:

    - uses BERT's pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of a
      punctuation character will be treated separately.

    - Such pretokenized input is BPE subtokenized

    This tokenizer inherits from :class:`~transformers.XLMTokenizer` which contains most of the methods. Users should
    refer to the superclass for more information regarding methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        tokenizer_file=None,
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sep_token="</s>",
        do_lowercase_and_remove_accent=False,
        **kwargs
    ):

        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=None,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            do_lowercase_and_remove_accent=do_lowercase_and_remove_accent,
            **kwargs,
        )
        self.bert_pre_tokenizer = BasicTokenizer(
            do_lower_case=False,
            never_split=self.all_special_tokens,
            tokenize_chinese_chars=False,
            strip_accents=False,
        )

    def _tokenize(self, text):

        pre_tokens = self.bert_pre_tokenizer.tokenize(text)

        split_tokens = []
        for token in pre_tokens:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens
