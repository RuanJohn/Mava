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

from typing import Tuple

import jaxmarl
import jumanji
import matrax
from jaxmarl.environments.smax import map_name_to_scenario
from jumanji.env import Environment
from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator as ConnectorGenerator,
)
from jumanji.environments.routing.lbf.generator import (
    RandomGenerator as LbfRandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.generator import (
    RandomGenerator as RwareRandomGenerator,
)
from jumanji.wrappers import AutoResetWrapper
from omegaconf import DictConfig

from mava.wrappers import (
    AgentIDWrapper,
    CentralControllerWrapper,
    GlobalStateWrapper,
    JaxMarlWrapper,
    LbfWrapper,
    MaConnectorWrapper,
    MatraxWrapper,
    RecordEpisodeMetrics,
    RwareWrapper,
)

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry = {
    "RobotWarehouse-v0": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging-v0": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
    "MaConnector-v2": {"generator": ConnectorGenerator, "wrapper": MaConnectorWrapper},
}

_matrax_registry = {
    "Matrax": {"wrapper": MatraxWrapper},
}


def add_optional_wrappers(
    env: Environment, config: DictConfig, add_global_state: bool = False
) -> Environment:
    # Add the global state to observation.
    if add_global_state:
        env = GlobalStateWrapper(env)

    # Add agent id to observation.
    if config.system.add_agent_id:
        env = AgentIDWrapper(env, add_global_state)

    if config.system.central_controller:
        if config.system.add_agent_id or add_global_state:
            raise ValueError(
                "Central controller and agent id or global state cannot be used together."
            )
        env = CentralControllerWrapper(env)

    return env


def make_jumanji_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[Environment, Environment]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
        A tuple of the environments.
    """
    # Config generator and select the wrapper.
    generator = _jumanji_registry[env_name]["generator"]
    generator = generator(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[env_name]["wrapper"]

    # Create envs.
    env = jumanji.make(env_name, generator=generator, **config.env.kwargs)
    eval_env = jumanji.make(env_name, generator=generator, **config.env.kwargs)
    env, eval_env = wrapper(env), wrapper(eval_env)

    env = add_optional_wrappers(env, config, add_global_state)
    eval_env = add_optional_wrappers(eval_env, config, add_global_state)

    env = AutoResetWrapper(env)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_matrax_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[Environment, Environment]:
    """
     Create a Matrax environment.
    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.
    Returns:
        Tuple of Matrax environments.
    """

    task_name = config["env"]["scenario"]["task_name"]
    wrapper = _matrax_registry[env_name]["wrapper"]

    # Create envs.
    env = matrax.make(task_name)
    eval_env = matrax.make(task_name)
    env, eval_env = wrapper(env), wrapper(eval_env)

    env = add_optional_wrappers(env, config, add_global_state)
    eval_env = add_optional_wrappers(eval_env, config, add_global_state)

    env = AutoResetWrapper(env)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make_jaxmarl_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[Environment, Environment]:
    """
     Create a JAXMARL environment.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
        A JAXMARL environment.
    """

    kwargs = dict(config.env.kwargs)
    if "smax" in env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Create jaxmarl envs.
    env = JaxMarlWrapper(jaxmarl.make(env_name, **kwargs), add_global_state)
    eval_env = JaxMarlWrapper(jaxmarl.make(env_name, **kwargs), add_global_state)

    # Add optional wrappers.
    if config.system.add_agent_id:
        env = AgentIDWrapper(env, add_global_state)
        eval_env = AgentIDWrapper(eval_env, add_global_state)

    env = AutoResetWrapper(env)
    env = RecordEpisodeMetrics(env)

    return env, eval_env


def make(config: DictConfig, add_global_state: bool = False) -> Tuple[Environment, Environment]:
    """
    Create environments for training and evaluation..

    Args:
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
        A tuple of the environments.
    """
    env_name = config.env.env_name

    if env_name in _jumanji_registry:
        return make_jumanji_env(env_name, config, add_global_state)
    elif env_name in jaxmarl.registered_envs:
        return make_jaxmarl_env(env_name, config, add_global_state)
    elif env_name in _matrax_registry:
        return make_matrax_env(env_name, config, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
