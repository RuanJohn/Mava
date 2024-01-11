# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

import functools
from typing import Optional, Sequence, Tuple, Union

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from mava.types import (
    Observation,
    ObservationCentralController,
    RNNGlobalObservation,
    RNNObservation,
)


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class MLPTorso(nn.Module):
    """MLP for processing vector environment observations."""

    layer_sizes: Sequence[int] = (128, 128)
    activation: str = "relu"
    use_layer_norm: bool = False

    def setup(self) -> None:
        if self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "tanh":
            self.activation_fn = nn.tanh

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation

        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)

        return x


class TransformerBlock(nn.Module):
    num_heads: int
    key_size: int
    mlp_units: Sequence[int]
    w_init_scale: float
    name: Optional[str] = None
    """Initialises the transformer block module.

    Args:
        num_heads: number of independent attention heads (H).
        key_size: the size of keys (K) and queries (Q) used in the attention mechanism.
        mlp_units: sequence of MLP layers in the feedforward networks following self-attention.
        w_init_scale: scale to `VarianceScaling` weight initializer.
        model_size: optional size of the output embedding (D'). If None, defaults
            to the key size multiplied by the number of heads (K * H).
        name: optional name for this module.
    """

    def setup(self) -> None:
        self.w_init = nn.initializers.variance_scaling(
            self.w_init_scale, "fan_in", "truncated_normal"
        )
        self.model_size = self.key_size * self.num_heads

    @nn.compact
    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: Optional[chex.Array] = None,
        mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Computes in this order:
            - (optionally masked) MHA with queries, keys & values
            - skip connection
            - layer norm
            - MLP
            - skip connection
            - layer norm

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
            query: embeddings sequence used to compute queries; shape [..., T', D_q].
            key: embeddings sequence used to compute keys; shape [..., T, D_k].
            value: embeddings sequence used to compute values; shape [..., T, D_v].
            mask: optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
            A new sequence of embeddings, consisting of a projection of the
                attention-weighted value projections; shape [..., T', D'].
        """

        # Multi-head attention and residual connection
        mha = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=self.w_init,
            out_features=self.model_size,
        )
        h = mha(inputs_q=query, inputs_kv=key) + query
        h = nn.LayerNorm(use_scale=True, use_bias=True)(h)

        # MLP and residual connection
        for mlp_layer in self.mlp_units:
            out = nn.Dense(mlp_layer)(h)
            out = nn.relu(out)
        out = nn.Dense(self.model_size)(h)
        out = nn.relu(out) + h
        out = nn.LayerNorm(use_scale=True, use_bias=True)(out)

        return out


class TransformerTorso(nn.Module):
    num_blocks: int
    num_heads: int
    key_size: int
    mlp_units: Sequence[int]
    name: Optional[str] = None

    def setup(self) -> None:
        self.model_size = self.num_heads * self.key_size

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        # Shape names:
        # B: batch size
        # O: observation size
        # H: hidden/embedding size
        # (B, O)
        # (B, O + 1) -> (B, H)
        embeddings = nn.Dense(self.model_size)(observation)

        # (B, H) -> (B, H)
        for block_id in range(self.num_blocks):
            transformer_block = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                mlp_units=self.mlp_units,
                w_init_scale=2 / self.num_blocks,
                name=f"self_attention_block_{block_id}",
            )
            embeddings = transformer_block(query=embeddings, key=embeddings)

        x = nn.Dense(
            self.model_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embeddings)
        x = nn.relu(x)
        return x  # (B, H)


class RecActor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]
    # TODO: The hidden size should depend on the pre-torso layer size.
    pre_torso: MLPTorso = MLPTorso(layer_sizes=(128,))
    post_torso: MLPTorso = MLPTorso(layer_sizes=(128,))

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: Union[RNNObservation, RNNGlobalObservation],
    ) -> Tuple[chex.Array, distrax.Categorical]:
        """Forward pass."""
        observation, done = observation_done

        policy_embedding = self.pre_torso(observation.agents_view)
        policy_rnn_in = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN()(policy_hidden_state, policy_rnn_in)

        actor_output = self.post_torso(policy_embedding)
        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        return policy_hidden_state, pi


class RecCentralisedCritic(nn.Module):
    """Critic Network."""

    pre_torso: MLPTorso = MLPTorso(layer_sizes=(128,))
    post_torso: MLPTorso = MLPTorso(layer_sizes=(128,))

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: RNNGlobalObservation,
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        critic_embedding = self.pre_torso(observation.global_state)

        critic_rnn_in = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN()(critic_hidden_state, critic_rnn_in)

        critic = self.post_torso(critic_embedding)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return critic_hidden_state, jnp.squeeze(critic, axis=-1)


class RecCritic(nn.Module):
    """Critic Network."""

    pre_torso: MLPTorso = MLPTorso(layer_sizes=(128,))
    post_torso: MLPTorso = MLPTorso(layer_sizes=(128,))

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        critic_embedding = self.pre_torso(observation.agents_view)
        critic_rnn_in = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN()(critic_hidden_state, critic_rnn_in)

        critic = self.post_torso(critic_embedding)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return critic_hidden_state, jnp.squeeze(critic, axis=-1)


class FFActor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]
    torso: MLPTorso = MLPTorso()

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        actor_output = self.torso(x)
        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )
        actor_policy = distrax.Categorical(logits=masked_logits)

        return actor_policy


class FFCritic(nn.Module):
    """Critic Network."""

    torso: MLPTorso = MLPTorso()

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        """Forward pass."""

        critic_output = self.torso(observation.agents_view)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_output
        )

        return jnp.squeeze(critic_output, axis=-1)


class FFCentralActor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]
    torso: MLPTorso = MLPTorso()

    @nn.compact
    def __call__(self, observation: ObservationCentralController) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        actor_output = self.torso(x)
        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)

        masked_logits = jnp.where(
            observation.joint_action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )
        actor_policy = distrax.Categorical(logits=masked_logits)

        return actor_policy
