# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" TF 2.0 MPNet model. """


import math
import warnings

import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_mpnet import MPNetConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/mpnet-base"
_CONFIG_FOR_DOC = "MPNetConfig"
_TOKENIZER_FOR_DOC = "MPNetTokenizer"

TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/mpnet-base",
]


class TFMPNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MPNetConfig
    base_model_prefix = "mpnet"

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)


class TFMPNetEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.padding_idx = 1
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.embeddings_sum = tf.keras.layers.Add()
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        super().build(input_shape)

    def create_position_ids_from_input_ids(self, input_ids):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = tf.math.cumsum(mask, axis=1) * mask

        return incremental_indices + self.padding_idx

    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids=input_ids)
            else:
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )
                position_ids = tf.tile(input=position_ids, multiples=(input_shape[0], 1))

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        final_embeddings = self.embeddings_sum(inputs=[inputs_embeds, position_embeds])
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


# Copied from transformers.models.bert.modeling_tf_bert.TFBertPooler with Bert->MPNet
class TFMPNetPooler(tf.keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)

        return pooled_output


class TFMPNetSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="q"
        )
        self.k = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="k"
        )
        self.v = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="v"
        )
        self.o = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="o"
        )
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        batch_size = shape_list(hidden_states)[0]

        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q, batch_size)
        k = self.transpose_for_scores(k, batch_size)
        v = self.transpose_for_scores(v, batch_size)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(shape_list(k)[-1], attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        # Apply relative position embedding (precomputed in MPNetEncoder) if provided.
        if position_bias is not None:
            attention_scores += position_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        attention_probs = self.dropout(attention_probs, training=training)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        c = tf.matmul(attention_probs, v)
        c = tf.transpose(c, perm=[0, 2, 1, 3])
        c = tf.reshape(c, (batch_size, -1, self.all_head_size))
        o = self.o(c)

        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs


class TFMPNetAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attn = TFMPNetSelfAttention(config, name="attn")
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input_tensor, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        self_outputs = self.attn(
            input_tensor, attention_mask, head_mask, output_attentions, position_bias=position_bias, training=training
        )
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_tf_bert.TFBertIntermediate with Bert->MPNet
class TFMPNetIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.bert.modeling_tf_bert.TFBertOutput with Bert->MPNet
class TFMPNetOutput(tf.keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFMPNetLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFMPNetAttention(config, name="attention")
        self.intermediate = TFMPNetIntermediate(config, name="intermediate")
        self.out = TFMPNetOutput(config, name="output")

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, position_bias=position_bias, training=training
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.out(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + outputs  # add attentions if we output them

        return outputs


class TFMPNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.n_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.initializer_range = config.initializer_range

        self.layer = [TFMPNetLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

    def build(self, input_shape):
        with tf.name_scope("relative_attention_bias"):
            self.relative_attention_bias = self.add_weight(
                name="embeddings",
                shape=[self.relative_attention_num_buckets, self.n_heads],
                initializer=get_initializer(self.initializer_range),
            )

        return super().build(input_shape)

    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=False,
    ):
        position_bias = self.compute_position_bias(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions,
                position_bias=position_bias,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += tf.cast(tf.math.less(n, 0), dtype=relative_position.dtype) * num_buckets
        n = tf.math.abs(n)

        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tf.math.less(n, max_exact)

        val_if_large = max_exact + tf.cast(
            tf.math.log(n / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )

        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        ret += tf.where(is_small, n, val_if_large)
        return ret

    def compute_position_bias(self, x, position_ids=None):
        """Compute binned relative position bias"""
        input_shape = shape_list(x)
        qlen, klen = input_shape[1], input_shape[1]

        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = tf.range(qlen)[:, None]
            memory_position = tf.range(klen)[None, :]

        relative_position = memory_position - context_position  # shape (qlen, klen)

        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
        )
        values = tf.gather(self.relative_attention_bias, rp_bucket)  # shape (qlen, klen, num_heads)
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)  # shape (1, num_heads, qlen, klen)
        return values


@keras_serializable
class TFMPNetMainLayer(tf.keras.layers.Layer):
    config_class = MPNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.encoder = TFMPNetEncoder(config, name="encoder")
        self.pooler = TFMPNetPooler(config, name="pooler")
        # The embeddings must be the last declaration in order to follow the weights order
        self.embeddings = TFMPNetEmbeddings(config, name="embeddings")

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer.get_input_embeddings
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer.set_input_embeddings
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["attention_mask"] is None:
            inputs["attention_mask"] = tf.fill(input_shape, 1)

        embedding_output = self.embeddings(
            inputs["input_ids"],
            inputs["position_ids"],
            inputs["inputs_embeds"],
            training=inputs["training"],
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(inputs["attention_mask"], (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, embedding_output.dtype)
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.num_hidden_layers

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            inputs["head_mask"],
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not inputs["return_dict"]:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


MPNET_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensor in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "attention_mask": attention_mask})`

    Args:
        config (:class:`~transformers.MPNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

MPNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.MPNetTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple. This
            argument can be used in eager mode, in graph mode the value will always be set to True.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare MPNet Model transformer outputting raw hidden-states without any specific head on top.",
    MPNET_START_DOCSTRING,
)
class TFMPNetModel(TFMPNetPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
        **kwargs,
    ):
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mpnet(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        return outputs

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=hs,
            attentions=attns,
        )


class TFMPNetLMHead(tf.keras.layers.Layer):
    """MPNet head for masked and permuted language modeling"""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.act = get_tf_activation("gelu")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        self.bias = value["bias"]
        self.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # project back to size of vocabulary with bias
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


@add_start_docstrings("""MPNet Model with a `language modeling` head on top. """, MPNET_START_DOCSTRING)
class TFMPNetForMaskedLM(TFMPNetPreTrainedModel, TFMaskedLanguageModelingLoss):

    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        self.lm_head = TFMPNetLMHead(config, self.mpnet.embeddings, name="lm_head")

    def get_lm_head(self):
        return self.lm_head

    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mpnet(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], prediction_scores)

        if not inputs["return_dict"]:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForMaskedLM.serving_output
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFMaskedLMOutput(logits=output.logits, hidden_states=hs, attentions=attns)


class TFMPNetClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

    def call(self, features, training=False):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    MPNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForSequenceClassification(TFMPNetPreTrainedModel, TFSequenceClassificationLoss):

    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        self.classifier = TFMPNetClassificationHead(config, name="classifier")

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mpnet(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, training=training)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification.serving_output
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    """
    MPNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForMultipleChoice(TFMPNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {"input_ids": tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)}

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )

        if inputs["input_ids"] is not None:
            num_choices = shape_list(inputs["input_ids"])[1]
            seq_length = shape_list(inputs["input_ids"])[2]
        else:
            num_choices = shape_list(inputs["inputs_embeds"])[1]
            seq_length = shape_list(inputs["inputs_embeds"])[2]

        flat_input_ids = tf.reshape(inputs["input_ids"], (-1, seq_length)) if inputs["input_ids"] is not None else None
        flat_attention_mask = (
            tf.reshape(inputs["attention_mask"], (-1, seq_length)) if inputs["attention_mask"] is not None else None
        )
        flat_position_ids = (
            tf.reshape(inputs["position_ids"], (-1, seq_length)) if inputs["position_ids"] is not None else None
        )
        flat_inputs_embeds = (
            tf.reshape(inputs["inputs_embeds"], (-1, seq_length, shape_list(inputs["inputs_embeds"])[3]))
            if inputs["inputs_embeds"] is not None
            else None
        )
        outputs = self.mpnet(
            flat_input_ids,
            flat_attention_mask,
            flat_position_ids,
            inputs["head_mask"],
            flat_inputs_embeds,
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=inputs["training"])
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], reshaped_logits)

        if not inputs["return_dict"]:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForMultipleChoice.serving_output
    def serving_output(self, output: TFMultipleChoiceModelOutput) -> TFMultipleChoiceModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFMultipleChoiceModelOutput(logits=output.logits, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    """
       MPNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
       Named-Entity-Recognition (NER) tasks.
       """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForTokenClassification(TFMPNetPreTrainedModel, TFTokenClassificationLoss):

    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mpnet(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output, training=inputs["training"])
        logits = self.classifier(sequence_output)

        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForTokenClassification.serving_output
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTokenClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    """
    MPNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForQuestionAnswering(TFMPNetPreTrainedModel, TFQuestionAnsweringLoss):

    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        training=False,
        **kwargs,
    ):
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """

        inputs = input_processing(
            func=self.call,
            config=self.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_positions=start_positions,
            end_positions=end_positions,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.mpnet(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None

        if inputs["start_positions"] is not None and inputs["end_positions"] is not None:
            labels = {"start_position": inputs["start_positions"]}
            labels["end_position"] = inputs["end_positions"]
            loss = self.compute_loss(labels, (start_logits, end_logits))

        if not inputs["return_dict"]:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForQuestionAnswering.serving_output
    def serving_output(self, output: TFQuestionAnsweringModelOutput) -> TFQuestionAnsweringModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits, end_logits=output.end_logits, hidden_states=hs, attentions=attns
        )
