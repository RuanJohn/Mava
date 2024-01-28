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

from typing import Optional, Sequence, Tuple, Union

from mava.utils.networks import CNNTorso, MLPTorso, TransformerTorso

VALID_TORSOS = ["mlp", "transformer", "cnn"]


def _make_mlp_torso(
    layer_sizes: Sequence[int],
    activation: str,
    use_layer_norm: bool,
) -> MLPTorso:
    """Creates a feedforward torso network.
    Args:
        layer_sizes (Sequence[int]): Size of each layer in the torso.
        activation (str): Activation function to use.
        use_layer_norm (bool): Whether to use layer norm.
    """

    return MLPTorso(
        layer_sizes=layer_sizes,
        activation=activation,
        use_layer_norm=use_layer_norm,
    )


def _make_transformer_torso(
    num_blocks: int, num_heads: int, mlp_units: Sequence[int], key_size: int, split_over_heads: bool
) -> TransformerTorso:
    return TransformerTorso(
        num_blocks=num_blocks,
        num_heads=num_heads,
        mlp_units=mlp_units,
        key_size=key_size,
        split_over_heads=split_over_heads,
    )


def _make_cnn_torso(
    conv_n_channels: int,
    activation: str,
    max_timesteps: int,
    should_concatenate_step_count: bool,
) -> CNNTorso:
    """Creates a feedforward torso network.
    Args:
        torso_type (str): Type of torso to use. This could be MLP, CNN or Transformer.
        conv_n_channels (int): Number of channels in the convolutional layers.
    """

    return CNNTorso(
        conv_n_channels=conv_n_channels,
        activation=activation,
        max_timesteps=max_timesteps,
        should_concatenate_step_count=should_concatenate_step_count,
    )


def make_torso(
    torso_config: dict,
) -> Union[MLPTorso, CNNTorso, TransformerTorso]:
    """Parses torso configuration."""
    if torso_config["network_type"] == "mlp":
        return _make_mlp_torso(**torso_config["network_kwargs"])

    elif torso_config["network_type"] == "transformer":
        return _make_transformer_torso(**torso_config["network_kwargs"])

    elif torso_config["network_type"] == "cnn":
        return _make_cnn_torso(**torso_config["network_kwargs"])

    else:
        raise ValueError(f"Unsupported network type: {torso_config['network_type']}")


def make_network_torsos(
    actor_network_config: dict,
    critic_network_config: dict,
) -> Tuple[
    Union[MLPTorso, CNNTorso, TransformerTorso],
    Optional[Union[MLPTorso, CNNTorso, TransformerTorso]],
    Union[MLPTorso, CNNTorso, TransformerTorso],
    Optional[Union[MLPTorso, CNNTorso, TransformerTorso]],
]:
    """Creates torso networks."""
    actor_pre_torso = make_torso(actor_network_config["pre_torso"])
    actor_post_torso = (
        make_torso(actor_network_config["post_torso"])
        if actor_network_config["post_torso"] is not None
        else None
    )
    critic_pre_torso = make_torso(critic_network_config["pre_torso"])
    critic_post_torso = (
        make_torso(critic_network_config["post_torso"])
        if critic_network_config["post_torso"] is not None
        else None
    )

    return actor_pre_torso, actor_post_torso, critic_pre_torso, critic_post_torso
