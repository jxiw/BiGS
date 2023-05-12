# coding=utf-8
# Copyright 2021. All rights reserved.
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
""" Flax BiGS model. """

from functools import partial
from typing import Callable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jax.nn.initializers import normal
from jax.numpy.linalg import eigh
from jax.scipy.signal import convolve

from .configuration_bigs import BiGSConfig
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxMaskedLMOutput,
    FlaxNextSentencePredictorOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
    FlaxMultipleChoiceModelOutput,
)
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.utils import logging

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigs"
_CONFIG_FOR_DOC = "BiGSConfig"
_TOKENIZER_FOR_DOC = "BiGSTokenizer"
BiGS_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading, saving and converting weights from
    PyTorch models)

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module
    and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`~BiGSConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the
            model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on
            GPUs) and `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see
            [`~FlaxPreTrainedModel.to_fp16`] and [`~FlaxPreTrainedModel.to_bf16`].
"""
BiGS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`~BiGSConfiTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)
        token_type_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`numpy.ndarray` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

"""


@flax.struct.dataclass
class FlaxBiGSForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        prediction_logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    """

    prediction_logits: jnp.ndarray = None
    seq_relationship_logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None


def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return jnp.fft.irfft(out)[: u.shape[0]]


def log_step_initializer(dt_min=0.01, dt_max=1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
                jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init


def make_HiPPO(N):
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return nhippo, P, B


def make_DPLR_HiPPO(N):
    """ Diagonalize NPLR representation """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    # Check skew symmetry
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    # assert jnp.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HiPPO(N)

    def init_Lambda_real(key, shape):
        assert shape == (N,)
        return Lambda.real

    def init_Lambda_imag(key, shape):
        assert shape == (N,)
        return Lambda.imag

    def init_P(key, shape):
        assert shape == (N,)
        return P

    def init_B(key, shape):
        assert shape == (N,)
        return B

    return init_Lambda_real, init_Lambda_imag, init_P, init_B


def vandermonde(v, L, alpha):
    """
    Computes v @ Vandermonde(alpha, L)
    v, alpha: shape (N,)
    Returns: shape (L,)
    """
    V = alpha[:, jnp.newaxis] ** jnp.arange(L)  # Vandermonde matrix
    return (v[jnp.newaxis, :] @ V)[0]


def s4d_kernel(C, A, L, step):
    Abar, Bbar = discretize(A, 1.0, step)
    return vandermonde(C * Bbar, L, Abar).real


@partial(jax.jit, static_argnums=2)
def s4d_kernel_zoh(C, A, L, step):
    """ A version of the kernel for B=1 and ZOH """
    kernel_l = lambda l: (C * (jnp.exp(step * A) - 1) / A * jnp.exp(l * step * A)).sum()
    return jax.vmap(kernel_l)(jnp.arange(L)).ravel().real


def discretize(A, B, step, mode="zoh"):
    if mode == "bilinear":
        return (1 + step / 2 * A) / (1 - step / 2 * A), step * B / (1 - step / 2 * A)
    elif mode == "zoh":
        return jnp.exp(step * A), (jnp.exp(step * A) - 1) / A * B


def s4d_ssm(C, A, L, step):
    N = A.shape[0]
    Abar, Bbar = discretize(A, jnp.ones(N), step, mode="zoh")
    Abar = jnp.diag(Abar)
    Bbar = Bbar.reshape(N, 1)
    Cbar = C.reshape(1, N)
    return Abar, Bbar, Cbar


class S4dLayer(nn.Module):
    N: int
    l_max: int
    decode: bool = False
    scaling: str = "hippo"

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "A_re": 0.1,
        "A_im": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        # Learned Parameters
        if self.scaling == "inv":
            self.A_re = self.param("A_re", nn.initializers.constant(-0.5), (self.N,))
            def arange_initializer(scale):
                return lambda key, shape: (shape[-1] / scale) * (shape[-1] / (2 * jnp.arange(shape[-1]) + 1) - 1)
            self.A_im = self.param("A_im", arange_initializer(jnp.pi), (self.N,))
        elif self.scaling == "lin":
            self.A_re = self.param("A_re", nn.initializers.constant(-0.5), (self.N,))
            def arange_initializer(scale):
                return lambda key, shape: scale * jnp.ones(shape) * jnp.arange(shape[-1])
            self.A_im = self.param("A_im", arange_initializer(jnp.pi), (self.N,))
        else:
            # default scaling is hippo
            hippo_A_real_initializer, hippo_A_imag_initializer, _, _ = hippo_initializer(self.N)
            self.A_re = self.param("A_re", hippo_A_real_initializer, (self.N,))
            self.A_im = self.param("A_im", hippo_A_imag_initializer, (self.N,))
        self.A = jnp.clip(self.A_re, None, -1e-4) + 1j * self.A_im
        self.C = self.param("C", normal(stddev=.5 ** .5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = jnp.exp(
            self.param("log_step", log_step_initializer(), (1,))
        )
        if not self.decode:
            self.K = s4d_kernel_zoh(self.C, self.A, self.l_max, self.step)
        else:
            # FLAX code to ensure that we only compute discrete once during decoding.
            def init_discrete():
                return s4d_ssm(self.C, self.A, self.l_max, self.step)

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", jnp.zeros, (self.N,), jnp.complex64
            )

    def __call__(self, u):
        if not self.decode:
            return causal_convolution(u, self.K) + self.D * u
        else:
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


class FlaxBiGSLayer(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_ssm = self.config.num_ssm
        self.max_seq_length = self.config.max_position_embeddings
        self.pre_norm = self.config.pre_norm
        self.decode = self.config.decode
        self.scaling = self.config.scaling
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.fs4 = S4dLayer(N=self.num_ssm, l_max=self.max_seq_length, decode=self.decode, scaling=self.scaling)
        self.bs4 = S4dLayer(N=self.num_ssm, l_max=self.max_seq_length, decode=self.decode, scaling=self.scaling)

        self.dv = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        self.du_forward = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        self.du_backward = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        self.duc_forward = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        self.duc_backward = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        self.dol = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

        self.do = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(
            self,
            hidden_states,
            deterministic: bool = True
    ):

        # hidden_states should be max_seq_len * hidden size
        def apply_s4(hidden_states):
            v = nn.gelu(self.dv(hidden_states))
            u_forward = nn.gelu(self.du_forward(hidden_states))
            u_backward = nn.gelu(self.du_backward(jnp.flip(hidden_states, axis=0)))
            # s4 layers
            fs4_output = jax.vmap(self.fs4.__call__, in_axes=1, out_axes=1)(u_forward)
            bs4_output = jax.vmap(self.bs4.__call__, in_axes=1, out_axes=1)(u_backward)
            # instead of sum, we concat states
            uc_forward = self.duc_forward(fs4_output)
            uc_backward = jnp.flip(self.duc_backward(bs4_output), axis=0)
            output = self.do(nn.gelu(self.dol(uc_forward * uc_backward)) * v)
            return output

        if self.pre_norm:
            hidden_states = hidden_states + apply_s4(self.LayerNorm(hidden_states))
        else:
            hidden_states = self.LayerNorm(hidden_states + apply_s4(hidden_states))

        return hidden_states


class FlaxBiGSLayerCollection(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxBiGSLayer(config=self.config, name=str(i), dtype=self.dtype) for i in
            range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states,
            attention_mask,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = layer(
                hidden_states,
                deterministic=deterministic,
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


class FlaxBiGSEncoder(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxBiGSLayerCollection(self.config, dtype=self.dtype)

    def __call__(
            self,
            hidden_states,
            attention_mask,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxBiGSEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.token_type_embeddings = nn.Embed(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxBiGSPooler(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, pooling_mask):
        if self.config.pooler_type == "mean":
            avg_hidden_state = jnp.mean(hidden_states, axis=-2)
            avg_hidden_state = self.dense(avg_hidden_state)
        else:
            non_pad_hidden_states = hidden_states * pooling_mask[:, :, jnp.newaxis]
            avg_hidden_state = jnp.sum(non_pad_hidden_states, axis=-2) / jnp.sum(pooling_mask, axis=-1)[:, jnp.newaxis]
            avg_hidden_state = self.dense(avg_hidden_state)
        return nn.tanh(avg_hidden_state)


class FlaxBiGSModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBiGSEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBiGSEncoder(self.config, dtype=self.dtype)
        self.pooler = FlaxBiGSPooler(self.config, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids: Optional[np.ndarray] = None,
            position_ids: Optional[np.ndarray] = None,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):

        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        @partial(jax.vmap, in_axes=(0, 0, 0, 0), out_axes=0)
        def vectorize_BiGS(s_input_ids, s_attention_mask, s_token_type_ids, s_position_ids):
            # here we already hide the batch dimension
            hidden_states = self.embeddings(
                input_ids=s_input_ids, token_type_ids=s_token_type_ids, position_ids=s_position_ids,
                deterministic=deterministic
            )

            # hidden_states dimension is sentence length x hidden size
            outputs = self.encoder(
                hidden_states,
                s_attention_mask,
                deterministic=deterministic,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            return outputs

        outputs = vectorize_BiGS(input_ids, attention_mask, token_type_ids, position_ids)

        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states, attention_mask) if self.add_pooling_layer else None
        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states
        )


class FlaxBiGSPredictionHeadTransform(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


class FlaxBiGSLMPredictionHead(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transform = FlaxBiGSPredictionHeadTransform(self.config, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.transform(hidden_states)

        if shared_embedding is not None:
            hidden_states = self.decoder.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)

        bias = jnp.asarray(self.bias, self.dtype)
        hidden_states += bias
        return hidden_states


class FlaxBiGSOnlyMLMHead(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBiGSLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states


class FlaxBiGSPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BiGSConfig
    base_model_prefix = "BiGS"
    module_class: nn.Module = None

    def __init__(
            self, config: BiGSConfig,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            _do_init: bool = True,
            **kwargs
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        input_shape = (1, config.max_position_embeddings)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        return self.module.init(
            rngs, input_ids, attention_mask, token_type_ids, position_ids, return_dict=False
        )["params"]

    @add_start_docstrings_to_model_forward(BiGS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            params: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # init input tensors if not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(token_type_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


add_start_docstrings(
    "The bare BiGS Model transformer outputting raw hidden-states without any specific head on top.",
    BiGS_START_DOCSTRING,
)


class FlaxBiGSModel(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSModule


class FlaxBiGSForMaskedLMModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxBiGSModule(config=self.config, add_pooling_layer=False, dtype=self.dtype)
        self.cls = FlaxBiGSOnlyMLMHead(config=self.config, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.s4_bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings("""BiGS Model with a `language modeling` head on top for MLM training. """,
                      BiGS_START_DOCSTRING)
class FlaxBiGSForMaskedLM(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSForMaskedLMModule


class FlaxBiGSPreTrainingHeads(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBiGSLMPredictionHead(self.config, dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FlaxBiGSForPreTrainingModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxBiGSModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxBiGSPreTrainingHeads(config=self.config, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):

        # Model
        outputs = self.s4_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.tie_word_embeddings:
            shared_embedding = self.s4_bert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        hidden_states = outputs[0]
        pooled_output = outputs[1]

        prediction_scores, seq_relationship_score = self.cls(
            hidden_states, pooled_output, shared_embedding=shared_embedding
        )

        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        return FlaxBiGSForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
        )


@add_start_docstrings(
    """
    BiGS Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BiGS_START_DOCSTRING,
)
class FlaxBiGSForPreTraining(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSForPreTrainingModule


FLAX_BiGS_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    ```
"""

overwrite_call_docstring(
    FlaxBiGSForPreTraining,
    BiGS_INPUTS_DOCSTRING.format("batch_size, sequence_length") + FLAX_BiGS_FOR_PRETRAINING_DOCSTRING,
)
append_replace_return_docstrings(
    FlaxBiGSForPreTraining, output_type=FlaxBiGSForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
)


class FlaxBiGSOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, pooled_output):
        return self.seq_relationship(pooled_output)


class FlaxBiGSForNextSentencePredictionModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxBiGSModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxBiGSOnlyNSPHead(dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Model
        outputs = self.s4_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)

        if not return_dict:
            return (seq_relationship_scores,) + outputs[2:]

        return FlaxNextSentencePredictorOutput(
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """BiGS Model with a `next sentence prediction (classification)` head on top.""",
    BiGS_START_DOCSTRING,
)
class FlaxBiGSForNextSentencePrediction(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSForNextSentencePredictionModule


class FlaxBiGSForSequenceClassificationModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxBiGSModule(config=self.config, dtype=self.dtype)
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """
    BiGS Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BiGS_START_DOCSTRING,
)
class FlaxBiGSForSequenceClassification(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSForSequenceClassificationModule


class FlaxBiGSForTokenClassificationModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxBiGSModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """
    BiGS Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BiGS_START_DOCSTRING,
)
class FlaxBiGSForTokenClassification(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSForTokenClassificationModule


class FlaxBiGSForQuestionAnsweringModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxBiGSModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.s4_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]

        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """
    BiGS Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BiGS_START_DOCSTRING,
)
class FlaxBiGSForQuestionAnswering(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSForQuestionAnsweringModule


class FlaxBiGSForMultipleChoiceModule(nn.Module):
    config: BiGSConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.s4_bert = FlaxBiGSModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.classifier = nn.Dense(1, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        if len(input_ids.shape) == 2:
            # in the initialization
            num_choices = 1
        else:
            # in the training, input_ids is 3d
            num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        outputs = self.s4_bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        # reshaped_logits should be number of question in the batch x number of choices
        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states
        )


@add_start_docstrings(
    """
    BiGS Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BiGS_START_DOCSTRING,
)
class FlaxBiGSForMultipleChoice(FlaxBiGSPreTrainedModel):
    module_class = FlaxBiGSForMultipleChoiceModule
