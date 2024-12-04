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

from carbs import CARBSParams, LinearSpace, LogitSpace, LogSpace, Param

param_spaces = [
    Param(name="actor_lr", space=LogSpace(scale=0.5, min=1e-6, max=1e-3), search_center=1e-4),
    Param(name="critic_lr", space=LogSpace(scale=0.5, min=1e-6, max=1e-3), search_center=1e-4),
    Param(name="ppo_epochs", space=LinearSpace(is_integer=True, min=1, max=20), search_center=4),
    Param(
        name="num_minibatches", space=LinearSpace(is_integer=True, min=1, max=4), search_center=2
    ),
    Param(name="gamma", space=LogitSpace(min=0.01, max=1.0), search_center=0.99),
    Param(name="gae_lambda", space=LogitSpace(min=0.01, max=1.0), search_center=0.95),
    Param(name="clip_eps", space=LogitSpace(min=0.01, max=0.7), search_center=0.2),
    Param(name="ent_coef", space=LogSpace(min=0.0, max=10.0), search_center=0.01),
    Param(name="vf_coef", space=LogSpace(min=0.01, max=10.0), search_center=0.5),
    Param(name="max_grad_norm", space=LogSpace(min=0.01, max=20.0), search_center=0.5),
    Param(
        name="num_updates", space=LinearSpace(is_integer=True, min=16, max=244), search_center=40
    ),
    Param(name="decay_kappa", space=LogitSpace(min=0.01, max=1.0), search_center=0.2),
]
carbs_params = CARBSParams(
    better_direction_sign=1,
    is_wandb_logging_enabled=False,
    resample_frequency=0,
)
