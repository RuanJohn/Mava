import jax.numpy as jnp


def action_combinations(num_agents, num_actions):
    # Get Cartesian product using indices function
    grid = jnp.indices([num_actions] * num_agents).reshape(num_agents, -1).T
    return grid


## Pipes, is correct but is slow due to loops.
# def joint_action_mask(action_mask, combinations):
#     # Convert combinations to list for indexing
#     comb_list = combinations.tolist()

#     # Check each combination against the action mask
#     mask_values = []
#     for comb in comb_list:
#         mask_values.append(
#             all([action_mask[agent][comb[agent]] for agent in range(len(action_mask))])
#         )

#     joint_mask = jnp.array(mask_values)

#     return joint_mask

## Pipes but has bug
# def joint_action_mask(action_mask, combinations):
#     # Reshape action mask for broadcasting
#     reshaped_mask = jnp.array(action_mask)[:, :, None]

#     # Check if each combination is a valid action
#     valid_actions = reshaped_mask == combinations.T[:, None, :]

#     # Check that all actions in the combination are valid
#     joint_mask = jnp.all(jnp.all(valid_actions, axis=0), axis=0)

#     return joint_mask


def joint_action_mask(action_mask, combinations):
    # Use the action mask to determine if each combination's action is valid for each agent

    valid_actions = jnp.take_along_axis(action_mask, combinations.T, axis=1).T

    # Check that all actions in the combination are valid
    joint_mask = jnp.all(valid_actions, axis=1)

    return joint_mask
