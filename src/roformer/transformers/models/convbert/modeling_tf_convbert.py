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
""" TF 2.0 ConvBERT model. """


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
    TFSequenceSummary,
    TFTokenClassificationLoss,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_convbert import ConvBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "YituTech/conv-bert-base"
_CONFIG_FOR_DOC = "ConvBertConfig"
_TOKENIZER_FOR_DOC = "ConvBertTokenizer"

TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "YituTech/conv-bert-base",
    "YituTech/conv-bert-medium-small",
    "YituTech/conv-bert-small",
    # See all ConvBERT models at https://huggingface.co/models?filter=convbert
]


# Copied from transformers.models.albert.modeling_tf_albert.TFAlbertEmbeddings with Albert->ConvBert
class TFConvBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: ConvBertConfig, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.embedding_size = config.embedding_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.embeddings_sum = tf.keras.layers.Add()
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.type_vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertEmbeddings.call
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = self.embeddings_sum(inputs=[inputs_embeds, position_embeds, token_type_embeds])
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings


class TFConvBertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        new_num_attention_heads = int(config.num_attention_heads / config.head_ratio)
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            num_attention_heads = 1
        else:
            num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio

        self.num_attention_heads = num_attention_heads
        self.conv_kernel_size = config.conv_kernel_size

        assert (
            config.hidden_size % self.num_attention_heads == 0
        ), "hidden_size should be divisible by num_attention_heads"

        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        self.key_conv_attn_layer = tf.keras.layers.SeparableConv1D(
            self.all_head_size,
            self.conv_kernel_size,
            padding="same",
            activation=None,
            depthwise_initializer=get_initializer(1 / self.conv_kernel_size),
            pointwise_initializer=get_initializer(config.initializer_range),
            name="key_conv_attn_layer",
        )

        self.conv_kernel_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.conv_kernel_size,
            activation=None,
            name="conv_kernel_layer",
            kernel_initializer=get_initializer(config.initializer_range),
        )

        self.conv_out_layer = tf.keras.layers.Dense(
            self.all_head_size,
            activation=None,
            name="conv_out_layer",
            kernel_initializer=get_initializer(config.initializer_range),
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        conv_attn_layer = tf.multiply(mixed_key_conv_attn_layer, mixed_query_layer)

        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = tf.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        conv_kernel_layer = tf.nn.softmax(conv_kernel_layer, axis=1)

        paddings = tf.constant(
            [
                [
                    0,
                    0,
                ],
                [int((self.conv_kernel_size - 1) / 2), int((self.conv_kernel_size - 1) / 2)],
                [0, 0],
            ]
        )

        conv_out_layer = self.conv_out_layer(hidden_states)
        conv_out_layer = tf.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        conv_out_layer = tf.pad(conv_out_layer, paddings, "CONSTANT")

        unfold_conv_out_layer = tf.stack(
            [
                tf.slice(conv_out_layer, [0, i, 0], [batch_size, shape_list(mixed_query_layer)[1], self.all_head_size])
                for i in range(self.conv_kernel_size)
            ],
            axis=-1,
        )

        conv_out_layer = tf.reshape(unfold_conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])

        conv_out_layer = tf.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = tf.reshape(conv_out_layer, [-1, self.all_head_size])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], attention_scores.dtype)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        value_layer = tf.reshape(
            mixed_value_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size]
        )
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        conv_out = tf.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        context_layer = tf.concat([context_layer, conv_out], 2)
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.head_ratio * self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TFConvBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TFConvBertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFConvBertSelfAttention(config, name="self")
        self.dense_output = TFConvBertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        self_outputs = self.self_attention(
            input_tensor, attention_mask, head_mask, output_attentions, training=training
        )
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class GroupedLinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, num_groups, kernel_initializer, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.kernel_initializer = kernel_initializer
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.group_out_dim, self.group_in_dim, self.num_groups],
            initializer=self.kernel_initializer,
            trainable=True,
        )

        self.bias = self.add_weight(
            "bias", shape=[self.output_size], initializer=self.kernel_initializer, dtype=self.dtype, trainable=True
        )

    def call(self, hidden_states):
        batch_size = shape_list(hidden_states)[0]
        x = tf.transpose(tf.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim]), [1, 0, 2])
        x = tf.matmul(x, tf.transpose(self.kernel, [2, 1, 0]))
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [batch_size, -1, self.output_size])
        x = tf.nn.bias_add(value=x, bias=self.bias)
        return x


class TFConvBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.num_groups == 1:
            self.dense = tf.keras.layers.Dense(
                config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
            )
        else:
            self.dense = GroupedLinearLayer(
                config.hidden_size,
                config.intermediate_size,
                num_groups=config.num_groups,
                kernel_initializer=get_initializer(config.initializer_range),
                name="dense",
            )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TFConvBertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.num_groups == 1:
            self.dense = tf.keras.layers.Dense(
                config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
            )
        else:
            self.dense = GroupedLinearLayer(
                config.intermediate_size,
                config.hidden_size,
                num_groups=config.num_groups,
                kernel_initializer=get_initializer(config.initializer_range),
                name="dense",
            )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TFConvBertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFConvBertAttention(config, name="attention")
        self.intermediate = TFConvBertIntermediate(config, name="intermediate")
        self.bert_output = TFConvBertOutput(config, name="output")

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


class TFConvBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.layer = [TFConvBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

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
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], output_attentions, training=training
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


class TFConvBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.embedding_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


@keras_serializable
class TFConvBertMainLayer(tf.keras.layers.Layer):
    config_class = ConvBertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.embeddings = TFConvBertEmbeddings(config, name="embeddings")

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = tf.keras.layers.Dense(config.hidden_size, name="embeddings_project")

        self.encoder = TFConvBertEncoder(config, name="encoder")
        self.config = config

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return head_mask

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
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
            token_type_ids=token_type_ids,
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

        if inputs["token_type_ids"] is None:
            inputs["token_type_ids"] = tf.fill(input_shape, 0)

        hidden_states = self.embeddings(
            inputs["input_ids"],
            inputs["position_ids"],
            inputs["token_type_ids"],
            inputs["inputs_embeds"],
            training=inputs["training"],
        )
        extended_attention_mask = self.get_extended_attention_mask(
            inputs["attention_mask"], input_shape, hidden_states.dtype
        )
        inputs["head_mask"] = self.get_head_mask(inputs["head_mask"])

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states, training=inputs["training"])

        hidden_states = self.encoder(
            hidden_states,
            extended_attention_mask,
            inputs["head_mask"],
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            inputs["return_dict"],
            training=inputs["training"],
        )

        return hidden_states


class TFConvBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvBertConfig
    base_model_prefix = "convbert"


CONVBERT_START_DOCSTRING = r"""

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

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Args:
        config (:class:`~transformers.ConvBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

CONVBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.ConvBertTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
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
    "The bare ConvBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CONVBERT_START_DOCSTRING,
)
class TFConvBertModel(TFConvBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.convbert = TFConvBertMainLayer(config, name="convbert")

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        token_type_ids=None,
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
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            kwargs_call=kwargs,
        )
        outputs = self.convbert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )

        return outputs

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutput(last_hidden_state=output.last_hidden_state, hidden_states=hs, attentions=attns)


class TFConvBertMaskedLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self):
        return self.input_embeddings

    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {"bias": self.bias}

    def set_bias(self, value):
        self.bias = value["bias"]
        self.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states):
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


class TFConvBertGeneratorPredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dense = tf.keras.layers.Dense(config.embedding_size, name="dense")

    def call(self, generator_hidden_states, training=False):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_tf_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


@add_start_docstrings("""ConvBERT Model with a `language modeling` head on top. """, CONVBERT_START_DOCSTRING)
class TFConvBertForMaskedLM(TFConvBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, **kwargs)

        self.vocab_size = config.vocab_size
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        self.generator_predictions = TFConvBertGeneratorPredictions(config, name="generator_predictions")

        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act

        self.generator_lm_head = TFConvBertMaskedLMHead(config, self.convbert.embeddings, name="generator_lm_head")

    def get_lm_head(self):
        return self.generator_lm_head

    def get_prefix_bias_name(self):
        return self.name + "/" + self.generator_lm_head.name

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        token_type_ids=None,
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
            token_type_ids=token_type_ids,
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
        generator_hidden_states = self.convbert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output, training=inputs["training"])
        prediction_scores = self.generator_lm_head(prediction_scores, training=inputs["training"])
        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], prediction_scores)

        if not inputs["return_dict"]:
            output = (prediction_scores,) + generator_hidden_states[1:]

            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )

    # Copied from transformers.models.bert.modeling_tf_bert.TFBertForMaskedLM.serving_output
    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFMaskedLMOutput(logits=output.logits, hidden_states=hs, attentions=attns)


class TFConvBertClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

        self.config = config

    def call(self, hidden_states, **kwargs):
        x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_tf_activation(self.config.hidden_act)(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


@add_start_docstrings(
    """
    ConvBERT Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForSequenceClassification(TFConvBertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        self.classifier = TFConvBertClassificationHead(config, name="classifier")

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        token_type_ids=None,
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
            token_type_ids=token_type_ids,
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
        outputs = self.convbert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            head_mask=inputs["head_mask"],
            inputs_embeds=inputs["inputs_embeds"],
            output_attentions=inputs["output_attentions"],
            output_hidden_states=inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        logits = self.classifier(outputs[0], training=inputs["training"])
        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], logits)

        if not inputs["return_dict"]:
            output = (logits,) + outputs[1:]

            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFSequenceClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    """
    ConvBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForMultipleChoice(TFConvBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.convbert = TFConvBertMainLayer(config, name="convbert")
        self.sequence_summary = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="sequence_summary"
        )
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
        return {"input_ids": tf.convert_to_tensor(MULTIPLE_CHOICE_DUMMY_INPUTS)}

    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
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
        token_type_ids=None,
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
            token_type_ids=token_type_ids,
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
        flat_token_type_ids = (
            tf.reshape(inputs["token_type_ids"], (-1, seq_length)) if inputs["token_type_ids"] is not None else None
        )
        flat_position_ids = (
            tf.reshape(inputs["position_ids"], (-1, seq_length)) if inputs["position_ids"] is not None else None
        )
        flat_inputs_embeds = (
            tf.reshape(inputs["inputs_embeds"], (-1, seq_length, shape_list(inputs["inputs_embeds"])[3]))
            if inputs["inputs_embeds"] is not None
            else None
        )
        outputs = self.convbert(
            flat_input_ids,
            flat_attention_mask,
            flat_token_type_ids,
            flat_position_ids,
            inputs["head_mask"],
            flat_inputs_embeds,
            inputs["output_attentions"],
            inputs["output_hidden_states"],
            return_dict=inputs["return_dict"],
            training=inputs["training"],
        )
        logits = self.sequence_summary(outputs[0], training=inputs["training"])
        logits = self.classifier(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if inputs["labels"] is None else self.compute_loss(inputs["labels"], reshaped_logits)

        if not inputs["return_dict"]:
            output = (reshaped_logits,) + outputs[1:]

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
                "token_type_ids": tf.TensorSpec((None, None, None), tf.int32, name="token_type_ids"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFMultipleChoiceModelOutput(logits=output.logits, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    """
    ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForTokenClassification(TFConvBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = tf.keras.layers.Dropout(classifier_dropout)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        token_type_ids=None,
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
            token_type_ids=token_type_ids,
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
        outputs = self.convbert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
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

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFTokenClassifierOutput(logits=output.logits, hidden_states=hs, attentions=attns)


@add_start_docstrings(
    """
    ConvBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CONVBERT_START_DOCSTRING,
)
class TFConvBertForQuestionAnswering(TFConvBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.convbert = TFConvBertMainLayer(config, name="convbert")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        token_type_ids=None,
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
            token_type_ids=token_type_ids,
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
        outputs = self.convbert(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
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
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def serving_output(self, output):
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits, end_logits=output.end_logits, hidden_states=hs, attentions=attns
        )
