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

"""Subprocess worker for GPU memory benchmarking.

Usage:
    python _memory_runner.py --module mava.systems.ppo.anakin.ff_ippo \
        --config-name ff_ippo.yaml \
        env=rware scenario=tiny-2ag system.num_updates=8 ...
"""

import argparse
import importlib
import os

import jax
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module", required=True, help="Python module path, e.g. mava.systems.ppo.anakin.ff_ippo"
    )
    parser.add_argument("--config-name", required=True, help="Hydra config name, e.g. ff_ippo.yaml")
    args, overrides = parser.parse_known_args()

    # Resolve absolute config dir path.
    config_dir = os.path.join(os.getcwd(), "mava", "configs", "default")

    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    # Allow dynamic attributes (matches hydra_entry_point behaviour).
    OmegaConf.set_struct(cfg, False)

    # Import the system module and call run_experiment.
    module = importlib.import_module(args.module)
    module.run_experiment(cfg)

    # Query peak GPU memory after training.
    device = jax.local_devices()[0]
    mem_stats = device.memory_stats()
    if mem_stats is not None:
        peak_bytes = mem_stats["peak_bytes_in_use"]
    else:
        # Fallback: memory_stats not available (CPU-only).
        peak_bytes = -1

    print(f"PEAK_MEM_BYTES:{peak_bytes}", flush=True)


if __name__ == "__main__":
    main()
