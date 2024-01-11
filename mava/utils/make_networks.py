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

from typing import Sequence, Tuple, Union

from mava.utils.networks import MLPTorso


def _make_mlp_torso(
    torso_type: str,
    layer_sizes: Sequence[int],
    activation: str,
    use_layer_norm: bool,
) -> MLPTorso:
    """Creates a feedforward torso network.
    Args:
        torso_type (str): Type of torso to use. This could be MLP, CNN or Transformer.
        layer_sizes (Sequence[int]): Size of each layer in the torso.
        activation (str): Activation function to use.
        use_layer_norm (bool): Whether to use layer norm.
    """

    if torso_type == "mlp":
        return MLPTorso(
            layer_sizes=layer_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
    else:
        raise ValueError(f"Unsupported torso type: {torso_type}")


def make_network_torsos(
    network_config: dict,
) -> Union[MLPTorso, Tuple[MLPTorso, MLPTorso]]:
    """Creates a torso network."""
    if network_config["network_type"] == "feedforward":
        return _make_mlp_torso(**network_config["pre_torso_kwargs"])

    elif network_config["network_type"] == "recurrent":
        return (
            _make_mlp_torso(**network_config["pre_torso_kwargs"]),
            _make_mlp_torso(**network_config["post_torso_kwargs"]),
        )

    else:
        raise ValueError(f"Unsupported network type: {network_config['network_type']}")
