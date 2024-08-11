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

import copy
import time
from typing import Any, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint

from mava.evaluator import get_eval_fn, make_indep_ff_tabular_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.networks import TabularPolicy
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import ActorApply, CriticApply, ExperimentOutput, LearnerFn, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.total_timestep_checker import check_total_timesteps
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def get_actions_log_prob(
    params: FrozenDict, apply_fn: ActorApply, key: chex.PRNGKey, num_agents: int, batch_size: int
) -> Tuple[chex.Array, chex.Array]:
    """Get actions and log probabilities from the agents."""

    agent_actions = jnp.zeros((batch_size, num_agents), dtype=jnp.int32)
    agent_log_probs = jnp.zeros((batch_size, num_agents), dtype=jnp.float32)

    for agent_idx in range(num_agents):
        key, act_key = jax.random.split(key)
        agent_params = tree.map(lambda x, agent_idx=agent_idx: x[agent_idx], params)
        agent_policy = apply_fn(agent_params, batch_size)
        agent_action = agent_policy.sample(seed=act_key)
        agent_log_prob = agent_policy.log_prob(agent_action)
        agent_actions = agent_actions.at[:, agent_idx].set(agent_action)
        agent_log_probs = agent_log_probs.at[:, agent_idx].set(agent_log_prob)

    return agent_actions, agent_log_probs


def get_log_probs_entropies(
    params: FrozenDict,
    apply_fn: ActorApply,
    key: chex.PRNGKey,
    num_agents: int,
    batch_size: int,
    actions: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Get log probabilities anfd policy entropy from the agents."""

    agent_log_probs = jnp.zeros((batch_size, num_agents), dtype=jnp.float32)
    agent_entropies = jnp.zeros((batch_size, num_agents), dtype=jnp.float32)

    for agent_idx in range(num_agents):
        key, entropy_key = jax.random.split(key)
        agent_params = tree.map(lambda x, agent_idx=agent_idx: x[agent_idx], params)
        agent_policy = apply_fn(agent_params, batch_size)
        agent_log_prob = agent_policy.log_prob(actions[:, agent_idx])
        agent_entropy = agent_policy.entropy(seed=entropy_key)
        agent_log_probs = agent_log_probs.at[:, agent_idx].set(agent_log_prob)
        agent_entropies = agent_entropies.at[:, agent_idx].set(agent_entropy)

    return agent_log_probs, agent_entropies


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""
    # Get apply and update functions for actor and critic networks.
    actor_apply_fn = apply_fns
    actor_update_fn = update_fns

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.

        """

        def _env_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, PPOTransition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            batch_size = last_timestep.observation.agents_view.shape[0]

            action, log_prob = get_actions_log_prob(
                params.actor_params,
                actor_apply_fn,
                policy_key,
                config.system.num_agents,
                batch_size,
            )

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            info = timestep.extras["episode_metrics"]

            transition = PPOTransition(
                None,
                action,
                None,
                timestep.reward,
                log_prob,
                None,
                info,
            )
            learner_state = LearnerState(params, opt_states, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        params, opt_states, key, env_state, last_timestep = learner_state

        # Broadcast the reward
        advantages, targets = traj_batch.reward - jnp.array(config.env.mean_payoff), None

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptState,
                    traj_batch: PPOTransition,
                    advantage: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    batch_size = traj_batch.action.shape[0]
                    # actor_policy = actor_apply_fn(actor_params, batch_size)
                    # log_prob = actor_policy.log_prob(traj_batch.action)
                    log_prob, entropy = get_log_probs_entropies(
                        actor_params,
                        actor_apply_fn,
                        key,
                        config.system.num_agents,
                        batch_size,
                        traj_batch.action,
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    loss_actor1 = ratio * advantage
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * advantage
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy.mean()
                    return total_loss_actor, (loss_actor, entropy)

                # CALCULATE ACTOR LOSS
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    opt_states.actor_opt_state,
                    traj_batch,
                    advantages,
                    entropy_key,
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = jax.vmap(actor_update_fn)(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = jax.vmap(optax.apply_updates)(params.actor_params, actor_updates)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = Params(actor_new_params, params.critic_params)
                new_opt_state = OptStates(actor_new_opt_state, opt_states.critic_opt_state)

                # PACK LOSS INFO
                total_loss = actor_loss_info[0]
                actor_loss = actor_loss_info[1][0]
                entropy = actor_loss_info[1][1]
                loss_info = {
                    "total_loss": total_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents
    # NOTE: This will only work for generalised matrax.
    config.env.mean_payoff = float(jnp.mean(env.payoff_matrix))

    # PRNG keys.
    key, actor_net_key, _ = keys

    # Define network and optimiser.
    actor_network = TabularPolicy(num_agent=1, num_actions=env.action_dim)

    actor_lr = make_learning_rate(config.system.actor_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )

    # Initialise actor params and optimiser state.
    # Init with batch size 1
    actor_net_keys = jax.random.split(actor_net_key, config.system.num_agents)

    actor_params = jax.vmap(actor_network.init, in_axes=(0, None))(actor_net_keys, 1)
    actor_opt_state = jax.vmap(actor_optim.init)(actor_params)

    # Pack params.
    params = Params(actor_params, None)

    # Pack apply and update functions.
    apply_fns = actor_network.apply
    update_fns = actor_optim.update

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, None)
    replicate_learner = (params, opt_states, step_keys)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, step_keys = replicate_learner
    init_learner_state = LearnerState(params, opt_states, step_keys, env_states, timesteps)

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_indep_ff_tabular_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../configs",
    config_name="default_ff_ippo_tabular_split.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
