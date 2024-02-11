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
from typing import Callable, Optional, Sequence, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

from mava.types import (
    Observation,
    ObservationCentralController,
    ObservationGlobalState,
    RNNGlobalObservation,
    RNNObservation,
)


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation_fn: Callable[[chex.Array], chex.Array] = nn.relu
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(layer_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)
        return x


class FeedForwardActor(nn.Module):
    """Feedforward Actor Network."""

    torso: nn.Module
    num_actions: Sequence[int]

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        x = self.torso(x)
        actor_logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(x)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        return distrax.Categorical(logits=masked_logits)


class FeedForwardCritic(nn.Module):
    """Feedforward Critic Network."""

    torso: nn.Module
    centralised_critic: bool = False

    @nn.compact
    def __call__(self, observation: Union[Observation, ObservationGlobalState]) -> chex.Array:
        """Forward pass."""
        if self.centralised_critic:
            if not isinstance(observation, ObservationGlobalState):
                raise ValueError("Global state must be provided to the centralised critic.")
            # Get global state in the case of a centralised critic.
            observation = observation.global_state
        else:
            # Get single agent view in the case of a decentralised critic.
            observation = observation.agents_view

        critic_output = self.torso(observation)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_output)

        return jnp.squeeze(critic_output, axis=-1)


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
            resets[:, np.newaxis],
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


class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    action_dim: Sequence[int]
    pre_torso: nn.Module
    post_torso: nn.Module

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, distrax.Categorical]:
        """Forward pass."""
        observation, done = observation_done

        policy_embedding = self.pre_torso(observation.agents_view)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN()(policy_hidden_state, policy_rnn_input)
        actor_logits = self.post_torso(policy_embedding)
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor_logits)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        return policy_hidden_state, pi


class RecurrentCritic(nn.Module):
    """Recurrent Critic Network."""

    pre_torso: nn.Module
    post_torso: nn.Module
    centralised_critic: bool = False

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: Union[RNNObservation, RNNGlobalObservation],
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        if self.centralised_critic:
            if not isinstance(observation, ObservationGlobalState):
                raise ValueError("Global state must be provided to the centralised critic.")
            # Get global state in the case of a centralised critic.
            observation = observation.global_state
        else:
            # Get single agent view in the case of a decentralised critic.
            observation = observation.agents_view

        critic_embedding = self.pre_torso(observation)
        critic_rnn_input = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN()(critic_hidden_state, critic_rnn_input)
        critic_output = self.post_torso(critic_embedding)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0))(critic_output)

        return critic_hidden_state, jnp.squeeze(critic_output, axis=-1)


class FeedForwardCentralActor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]
    torso: MLPTorso = MLPTorso(
        layer_sizes=[128, 128],
        activation_fn=nn.relu,
    )

    @nn.compact
    def __call__(self, observation: ObservationCentralController) -> distrax.Categorical:
        """Forward pass."""
        # x = observation.agents_view

        actor_output = self.torso(observation)
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


class TransformerBlock(nn.Module):
    num_heads: int
    key_size: int
    mlp_units: Sequence[int]
    w_init_scale: float
    split_over_heads: bool
    name: Optional[str] = None
    """Initialises the transformer block module.
    Args:
        num_heads: number of independent attention heads (H).
        key_size: the size of keys (K) and queries (Q) used in the attention mechanism.
        mlp_units: sequence of MLP layers in the feedforward networks following self-attention.
        w_init_scale: scale to `VarianceScaling` weight initializer.
        split_over_heads: whether to split keys, queries and values over heads.
        name: optional name for this module.
    """

    def setup(self) -> None:
        self.w_init = nn.initializers.variance_scaling(
            self.w_init_scale, "fan_in", "truncated_normal"
        )
        if self.split_over_heads:
            if self.key_size % self.num_heads != 0:
                raise ValueError("Key size must be divisible by number of heads")
            self.model_size = self.key_size
        else:
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
    split_over_heads: bool
    name: Optional[str] = None

    def setup(self) -> None:
        if self.split_over_heads:
            if self.key_size % self.num_heads != 0:
                raise ValueError("Key size must be divisible by number of heads")
            self.model_size = self.key_size
        else:
            self.model_size = self.key_size * self.num_heads

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        # Shape names:
        # B: batch size
        # O: observation size
        # H: hidden/embedding size
        # (B, O)
        # (B, O + 1) -> (B, H)
        embeddings = nn.Dense(self.model_size)(observation.agents_view)

        # (B, H) -> (B, H)
        for block_id in range(self.num_blocks):
            transformer_block = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                mlp_units=self.mlp_units,
                w_init_scale=2 / self.num_blocks,
                split_over_heads=self.split_over_heads,
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


def make_concatenate_step_count(
    should_concatenate_step_count: bool,
    max_timesteps: int,
) -> nn.Module:
    def normalize_step_count(step_count: chex.Array) -> chex.Array:
        return step_count / max_timesteps

    def concatenate_step_count(
        observation: chex.Array,
        step_count: chex.Array,
    ) -> chex.Array:
        if should_concatenate_step_count:
            step_count = normalize_step_count(step_count)
            return jnp.concatenate([observation, step_count], axis=-1)
        else:
            return observation

    return concatenate_step_count


class CNNTorso(nn.Module):
    """CNN for processing grid-based environment observations."""

    conv_n_channels: int = 32
    activation: str = "relu"
    max_timesteps: int = 1
    should_concatenate_step_count: bool = False

    def setup(self) -> None:
        if self.activation == "relu":
            self.activation_fn = nn.relu
        elif self.activation == "tanh":
            self.activation_fn = nn.tanh
        self.normalize_step_count = make_concatenate_step_count(
            should_concatenate_step_count=self.should_concatenate_step_count,
            max_timesteps=self.max_timesteps,
        )
        # Input will either be (batch, grid_size, grid_size)
        # or (batch, grid_size, grid_size, num_one_hot_features)
        self.cnn_block = nn.Sequential(
            [
                nn.Conv(features=self.conv_n_channels, kernel_size=(3, 3), padding="SAME"),
                self.activation_fn,
                nn.Conv(features=self.conv_n_channels, kernel_size=(3, 3), padding="SAME"),
                self.activation_fn,
                nn.Conv(features=self.conv_n_channels // 2, kernel_size=(3, 3), padding="SAME"),
                self.activation_fn,
                nn.Conv(features=2, kernel_size=(3, 3), padding="SAME"),
                self.activation_fn,
            ]
        )

    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation.agents_view  # (B, grid, grid) or (B, grid, grid, num_one_hot_features)
        x = self.cnn_block(x)  # (B, grid, grid, 2)
        x = x.reshape((x.shape[0], -1))  # (B, grid * grid * 2)

        x = self.normalize_step_count(x, jnp.expand_dims(observation.step_count, axis=-1))

        return x
