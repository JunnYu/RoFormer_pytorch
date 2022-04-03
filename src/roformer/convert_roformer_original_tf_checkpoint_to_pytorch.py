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
"""Convert RoFormer checkpoint."""

import argparse

import torch
from transformers.utils import logging

from roformer import RoFormerConfig, RoFormerForMaskedLM, RoFormerForCausalLM, load_tf_weights_in_roformer

logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(
    tf_checkpoint_path, bert_config_file, pytorch_dump_path, roformer_sim=False
):
    # Initialise PyTorch model
    config = RoFormerConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    
    if roformer_sim:
        # 如果转换roformer-sim的话，需要使用RoFormerForCausalLM，这个带有pooler的权重
        config.is_decoder = True
        config.eos_token_id = 102
        config.pooler_activation = "linear"
        model = RoFormerForCausalLM(config)
    else:
        model = RoFormerForMaskedLM(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_roformer(model, config, tf_checkpoint_path)

    # ignore 不保存roformer.encoder.embed_positions.weight
    _keys_to_ignore_on_save = ["roformer.encoder.embed_positions.weight"]
    state_dict = model.state_dict()
    for ignore_key in _keys_to_ignore_on_save:
        if ignore_key in state_dict.keys():
            del state_dict[ignore_key]
            
    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(
        state_dict, pytorch_dump_path, _use_new_zipfile_serialization=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--roformer_sim",
        action="store_true",
        help="Whether or not roformer-sim.",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path, args.roformer_sim
    )
