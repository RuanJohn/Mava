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

import chex
import jax.numpy as jnp


def get_all_action_combinations(num_agents: int, num_actions: int) -> chex.Array:
    # Get Cartesian product using indices function
    grid = jnp.indices([num_actions] * num_agents).reshape(num_agents, -1).T
    return grid


def compute_joint_action_mask(action_mask: chex.Array, combinations: chex.Array) -> chex.Array:
    # Expand dimensions to align batch and combination axes
    is_recurrent = len(action_mask.shape) == 4
    if is_recurrent:
        action_mask_expanded = action_mask[:, :, None, :, :]
        combinations_expanded = combinations[None, None, :, :, None]
        take_axis = 4
        agent_axis = 3
    else:
        action_mask_expanded = action_mask[:, None, :, :]
        combinations_expanded = combinations[None, :, :, None]
        take_axis = 3
        agent_axis = 2

    # Use the action mask to determine if each combination's action is valid for each agent
    valid_actions = jnp.take_along_axis(action_mask_expanded, combinations_expanded, axis=take_axis)

    # Check that all actions in the combination are valid
    joint_mask = jnp.all(valid_actions, axis=agent_axis)

    return joint_mask
