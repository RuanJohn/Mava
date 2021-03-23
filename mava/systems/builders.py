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

"""MARL system Builder interface."""

import abc
from typing import Dict, Iterator, Optional, Union

import reverb
import sonnet as snt
from acme import specs
from acme.utils import counting, loggers

from mava import adders, core

NestedLogger = Union[loggers.Logger, Dict[str, loggers.Logger]]


class SystemBuilder(abc.ABC):
    """Defines an interface for defining the components of an RL system.
    Implementations of this interface contain a complete specification of a
    concrete RL system. An instance of this class can be used to build an
    RL system which interacts with the environment either locally or in a
    distributed setup.
    """

    @abc.abstractmethod
    def make_replay_table(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> reverb.Table:
        """Create tables to insert data into."""

    @abc.abstractmethod
    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""

    @abc.abstractmethod
    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server.
        """

    @abc.abstractmethod
    def make_executor(
        self,
        policy_networks: Dict[str, snt.Module],
        adder: Optional[adders.Adder] = None,
        variable_clients: Optional[core.VariableSource] = None,
    ) -> core.Executor:
        """Create an executer instance.
        Args:
          policy_networks: A struct of instance of all the different
            policy networks; this should be a callable
            which takes as input observations and returns actions.
          adder: How data is recorded (e.g. added to replay).
          variable_clients: A source providing the necessary actor parameters.
        """

    @abc.abstractmethod
    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[NestedLogger] = None,
        # TODO: eliminate checkpoint and move it outside.
        checkpoint: bool = False,
    ) -> core.Trainer:
        """Creates an instance of the trainer.
        Args:
          networks: struct describing the networks needed by the trainer; this can
            be specific to the trainer in question.
          dataset: iterator over samples from replay.
          replay_client: client which allows communication with replay, e.g. in
            order to update priorities.
          counter: a Counter which allows for recording of counts (learner steps,
            actor steps, etc.) distributed throughout the agent.
          logger: Logger object for logging metadata.
          checkpoint: bool controlling whether the learner checkpoints itself.
        """
