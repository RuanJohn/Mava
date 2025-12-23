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


from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.networks.distributions import IdentityTransformation, TanhTransformedDistribution
from mava.utils.centralised_controller import compute_joint_action_mask, get_all_action_combinations


class TabularPolicy(nn.Module):
    num_agent: int
    num_actions: int

    def setup(self) -> None:
        self.policy_logits = self.param(
            "policy_logits", nn.initializers.zeros, (self.num_agent, self.num_actions)
        )

    def __call__(self, batch_size: int) -> tfd.Distribution:
        # Repeat the policy_logits for each item in the batch
        if self.num_agent > 1:
            batched_logits = jnp.repeat(self.policy_logits[None, :, :], batch_size, axis=0)
        else:
            batched_logits = jnp.repeat(self.policy_logits, batch_size, axis=0)

        return IdentityTransformation(distribution=tfd.Categorical(logits=batched_logits))

class FactoredTabularPolicy(nn.Module):
    num_agents: int
    num_actions: int

    def setup(self) -> None:

        self.joint_action_space = self.num_agents * self.num_actions
        # Initialize the logits for the factored tabular policy
        self.policy_logits = self.param(
            "policy_logits", nn.initializers.zeros, (1, self.joint_action_space)
        )

    def __call__(self, batch_size: int) -> tfd.Distribution:
        # Repeat the policy_logits for each item in the batch
        batched_logits = jnp.repeat(self.policy_logits, batch_size, axis=0)

        # Split the logits from (batch, num_actions) to (batch, num_agents, num_actions)
        batched_logits = batched_logits.reshape(batch_size, self.num_agents, self.num_actions)

        return IdentityTransformation(distribution=tfd.Categorical(logits=batched_logits))

class AutoregressiveTabularPolicy(nn.Module):
    num_agents: int
    num_actions: int

    def setup(self) -> None:
        N, A = self.num_agents, self.num_actions
        self.tables = []
        for i in range(N):
            self.tables.append(
                self.param(
                    f"policy_logits_step_{i}",
                    nn.initializers.zeros,
                    (A**i, A),  # π(a_i | a_<i) table
                )
            )

    def __call__(self, sample_key: chex.PRNGKey, batch_size: int) -> Tuple[chex.Array, chex.Array]:
        """
        Returns:
          actions: (B, N) int32
          logp:    (B,)   float32
        """
        N = self.num_agents
        A = jnp.int32(self.num_actions)
        B = batch_size

        # One RNG key per (agent step, batch element)
        keys = jax.random.split(sample_key, N * B).reshape(N, B, 2)

        def step(carry, inputs):
            idx, logp = carry                  # idx: (B,), logp: (B,)
            i, step_keys = inputs              # step_keys: (B, 2)

            table = self.tables[i]             # (A**i, A)
            logits = table[idx]                # (B, A)

            dist = tfd.Categorical(logits=logits)
            a = dist.sample(seed=step_keys)    # (B,)
            a = a.astype(jnp.int32)

            logp = logp + dist.log_prob(a)     # (B,)

            # base-A update for each batch element
            idx = idx * A + a                  # (B,)

            return (idx, logp), a

        init_idx = jnp.zeros((B,), dtype=jnp.int32)
        init_logp = jnp.zeros((B,), dtype=jnp.float32)

        (final_idx, final_logp), actions_TB = jax.lax.scan(
            step,
            (init_idx, init_logp),
            (jnp.arange(N), keys),
        )

        actions = jnp.transpose(actions_TB, (1, 0))  # (B, N)
        return actions, final_logp




class DiscreteActionHead(nn.Module):
    """Discrete Action Head"""

    action_dim: int
    is_central_controller: bool = False
    num_agents: Optional[int] = None
    num_indiv_actions: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        obs_embedding: chex.Array,
        action_mask: chex.Array,
    ) -> tfd.TransformedDistribution:
        """Action selection for distrete action space environments.

        Args:
        ----
            obs_embedding: Observation embedding from network torso.
            observation: Observation object containing `agents_view`, `action_mask` and
                `step_count`.

        Returns:
        -------
            A transformed tfd.categorical distribution on the action space for action sampling.

        NOTE: We pass both the observation embedding and the observation object to the action head
        since the observation object contains the action mask and other potentially useful
        information.

        """
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(obs_embedding)

        if self.is_central_controller:
            if self.num_agents is None or self.num_indiv_actions is None:
                raise ValueError(
                    "Number of agents and actions must be provided for central controller."
                )

            action_combinations = get_all_action_combinations(
                self.num_agents, self.num_indiv_actions
            )
            action_mask = compute_joint_action_mask(action_mask, action_combinations)
            action_mask = action_mask.squeeze(axis=-1)
        else:
            action_mask = action_mask

        masked_logits = jnp.where(
            action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        #  We transform this distribution with the `Identity()` transformation to
        # keep the API identical to the ContinuousActionHead.
        return IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))


class FactoredDiscreteActionHead(nn.Module):
    """Factored Discrete Action Head"""

    action_dim: int
    is_central_controller: bool = False
    num_agents: Optional[int] = None
    num_indiv_actions: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        obs_embedding: chex.Array,
        action_mask: chex.Array,
    ) -> tfd.TransformedDistribution:
        """Action selection for distrete action space environments.

        Args:
        ----
            obs_embedding: Observation embedding from network torso.
            observation: Observation object containing `agents_view`, `action_mask` and
                `step_count`.

        Returns:
        -------
            A transformed tfd.categorical distribution on the action space for action sampling.

        NOTE: We pass both the observation embedding and the observation object to the action head
        since the observation object contains the action mask and other potentially useful
        information.

        """
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(obs_embedding)

        # Reshape logits to factorize the action space
        new_shape = actor_logits.shape[:-1] + (self.num_agents, self.num_indiv_actions)
        actor_logits = actor_logits.reshape(new_shape)

        masked_logits = jnp.where(
            action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        #  We transform this distribution with the `Identity()` transformation to
        # keep the API identical to the ContinuousActionHead.
        return IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))


class ContinuousActionHead(nn.Module):
    """ContinuousActionHead using a transformed Normal distribution.

    Note: This network only handles the case where actions lie in the interval [-1, 1].
    """

    action_dim: int
    min_scale: float = 1e-3
    independent_std: bool = True  # whether or not the log_std is independent of the observation.

    # These are not needed but we keep them to keep the API consistent with the discrete case.
    is_central_controller: bool = False
    num_agents: Optional[int] = None
    num_indiv_actions: Optional[int] = None

    def setup(self) -> None:
        self.mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))

        if self.independent_std:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        else:
            self.log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))

    @nn.compact
    def __call__(self, obs_embedding: chex.Array, action_mask: chex.Array) -> tfd.Independent:
        """Action selection for continuous action space environments.

        Args:
        ----
            obs_embedding: Observation embedding.
            action_mask: Legal action mask for masked distributions. NOTE: In the
                continuous case, the action mask is not used but we still pass it in
                to keep the API consistent between the discrete and continuous cases.

        Returns:
        -------
            tfd.Independent: Independent transformed distribution.

        """
        del action_mask
        loc = self.mean(obs_embedding)

        scale = self.log_std if self.independent_std else self.log_std(obs_embedding)
        scale = jax.nn.softplus(scale) + self.min_scale

        distribution = tfd.Normal(loc=loc, scale=scale)

        return tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )
