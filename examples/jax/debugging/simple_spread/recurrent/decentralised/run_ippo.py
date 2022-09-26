# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

"""Example running IPPO on debug MPE environments."""
import functools
from datetime import datetime
from typing import Any

import optax
from absl import app, flags

from mava.systems.jax import ippo
from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "simple_spread",
    "Debugging environment name (str).",
)
flags.DEFINE_string(
    "action_space",
    "discrete",
    "Environment action space type (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def main(_: Any) -> None:
    """Example running recurrent IPPO on debugging environment."""

<<<<<<<< HEAD:examples/jax/debugging/simple_spread/recurrent/decentralised/run_ippo.py
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name=FLAGS.env_name,
        action_space=FLAGS.action_space,
    )
========
    # Environment
    environment_factory = functools.partial(make_environment, map_name=FLAGS.map_name)
>>>>>>>> origin/develop:examples/jax/smac/feedforward/decentralised/run_ippo_eval_intervals.py

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return ippo.make_default_networks(  # type: ignore
            policy_layer_sizes=(256, 256, 256),
            critic_layer_sizes=(512, 512, 256),
<<<<<<<< HEAD:examples/jax/debugging/simple_spread/recurrent/decentralised/run_ippo.py
            policy_recurrent_layer_sizes=(256,),
========
>>>>>>>> origin/develop:examples/jax/smac/feedforward/decentralised/run_ippo_eval_intervals.py
            *args,
            **kwargs,
        )

    # Used for checkpoints, tensorboard logging and env monitoring
    experiment_path = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    # Optimiser.
    policy_optimiser = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
<<<<<<<< HEAD:examples/jax/debugging/simple_spread/recurrent/decentralised/run_ippo.py
========
    )

    critic_optimiser = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
>>>>>>>> origin/develop:examples/jax/smac/feedforward/decentralised/run_ippo_eval_intervals.py
    )

    critic_optimiser = optax.chain(
        optax.clip_by_global_norm(40.0), optax.scale_by_adam(), optax.scale(-1e-4)
    )

    # Import the base IPPO system.
    system = ippo.IPPOSystem()

    # Update the system with the components necessary to make the policy recurrent.
    system.update(ippo.recurrent_policy_components)

    # Build the system.
    system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=experiment_path,
        policy_optimiser=policy_optimiser,
        critic_optimiser=critic_optimiser,
        run_evaluator=True,
        sample_batch_size=5,
        num_epochs=15,
        num_executors=1,
        multi_process=True,
<<<<<<<< HEAD:examples/jax/debugging/simple_spread/recurrent/decentralised/run_ippo.py
========
        evaluation_interval={"executor_steps": 10000},
        evaluation_duration={"evaluator_episodes": 32},
>>>>>>>> origin/develop:examples/jax/smac/feedforward/decentralised/run_ippo_eval_intervals.py
    )

    # Launch the system.
    system.launch()


if __name__ == "__main__":
    app.run(main)
