import time

import jax
import jax.numpy as jnp
import matrax

LR = 0.001
NUM_EPISODES = 10

# Instantiate a matrix game environment using the registry
env = matrax.make("Climbing-stateless-v0")

# Reset your (jit-able) environment
key = jax.random.PRNGKey(0)
key, env_key, agents_policy_key, action_selection_key = jax.random.split(key, 4)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Get agent action space specs
action_spec = env.action_spec()
observation_spec = env.observation_spec()

# Num agents
num_agents = env.num_agents
num_actions = action_spec.num_values[0]
observation_dim = observation_spec.agent_obs.shape[0]

# Create agent q-tables
q_tables = jax.random.uniform(
    key=agents_policy_key,
    shape=(num_agents, num_actions),
    dtype=jnp.float32,
)


def select_action(q_table, action_selection_key):
    # Select action using epsilon-greedy policy
    action_selection_key, key = jax.random.split(action_selection_key)
    select_random_action = jax.random.uniform(key) < 0.1
    action = jax.lax.cond(
        select_random_action,
        lambda: jax.random.randint(key, shape=(), minval=0, maxval=num_actions),
        lambda: jnp.argmax(q_table),
    )
    return action, action_selection_key


select_actions = jax.vmap(select_action, in_axes=(0, None), out_axes=(0, None))
select_actions = jax.jit(select_actions)


def update_q_table(
    q_table,
    action,
    reward,
    lr,
):
    q_table = q_table.at[action].set(q_table[action] + lr * (reward - q_table[action]))

    return q_table


update_q_tables = jax.vmap(update_q_table, in_axes=(0, 0, 0, None))
update_q_tables = jax.jit(update_q_tables)

for _ in range(NUM_EPISODES):
    env_key, env_key_use = jax.random.split(env_key)
    state, timestep = jit_reset(env_key_use)
    start_time = time.time()
    while not timestep.last():
        action, action_selection_key = select_actions(q_tables, action_selection_key)
        state, timestep = jit_step(
            state, action
        )  # Take a step and observe the next state and time step
        q_tables = update_q_tables(q_tables, action, timestep.reward, LR)
        # print(f"reward: {jnp.mean(timestep.reward)}, obs: {timestep.observation.agent_obs}")
    print(f"SPS: {500 / (time.time() - start_time)}")
    print(q_tables)
