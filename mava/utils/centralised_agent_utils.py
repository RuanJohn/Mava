import jax.numpy as jnp


def action_combinations(num_agents, num_actions):
    # Get Cartesian product using indices function
    grid = jnp.indices([num_actions] * num_agents).reshape(num_agents, -1).T
    return grid


def joint_action_mask(action_mask, combinations):
    # Use the action mask to determine if each combination's action is valid for each agent

    valid_actions = jnp.take_along_axis(action_mask, combinations.T, axis=1).T

    # Check that all actions in the combination are valid
    joint_mask = jnp.all(valid_actions, axis=1)

    return joint_mask
