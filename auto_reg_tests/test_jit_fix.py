#!/usr/bin/env python3
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

"""
Test script to verify AutoregressiveTabularPolicy works in jitted context.

Run with: source .venv/bin/activate && python test_jit_fix.py
"""

import jax

from mava.networks.heads import AutoregressiveTabularPolicy


def main():
    print("=" * 60)
    print("Testing AutoregressiveTabularPolicy JIT compatibility")
    print("=" * 60)

    # Setup
    num_agents = 2
    num_actions = 3
    batch_size = 4

    print("\nConfiguration:")
    print(f"  num_agents: {num_agents}")
    print(f"  num_actions: {num_actions}")
    print(f"  batch_size: {batch_size}")

    # Create and initialize network
    network = AutoregressiveTabularPolicy(num_agents=num_agents, num_actions=num_actions)

    key = jax.random.PRNGKey(0)
    init_key, _ = jax.random.split(key)
    params = network.init(init_key, init_key, batch_size=1, method="act_fn")

    # Define apply function
    def apply_fn(params, sample_key, batch_size):
        return network.apply(params, sample_key, batch_size, method="act_fn")

    # Test 1: Non-jitted
    print("\n[Test 1] Non-jitted execution...")
    sample_key = jax.random.PRNGKey(1)
    actions, logp = apply_fn(params, sample_key, batch_size)
    assert actions.shape == (
        batch_size,
        num_agents,
    ), f"Expected shape ({batch_size}, {num_agents}), got {actions.shape}"
    assert logp.shape == (batch_size,), f"Expected shape ({batch_size},), got {logp.shape}"
    print(f"  ✓ Actions shape: {actions.shape}")
    print(f"  ✓ Logp shape: {logp.shape}")

    # Test 2: Jitted (this is the critical test)
    print("\n[Test 2] Jitted execution (compilation + execution)...")
    # Make batch_size static since it's needed for key generation
    jitted_fn = jax.jit(apply_fn, static_argnums=2)

    sample_key = jax.random.PRNGKey(2)
    actions_jit, logp_jit = jitted_fn(params, sample_key, batch_size)

    assert actions_jit.shape == (
        batch_size,
        num_agents,
    ), f"Expected shape ({batch_size}, {num_agents}), got {actions_jit.shape}"
    assert logp_jit.shape == (batch_size,), f"Expected shape ({batch_size},), got {logp_jit.shape}"
    print(f"  ✓ Actions shape: {actions_jit.shape}")
    print(f"  ✓ Logp shape: {logp_jit.shape}")

    # Test 3: Multiple jitted calls
    print("\n[Test 3] Multiple jitted calls...")
    for i in range(3):
        sample_key = jax.random.PRNGKey(10 + i)
        actions_test, logp_test = jitted_fn(params, sample_key, batch_size)
        assert actions_test.shape == (batch_size, num_agents)
        assert logp_test.shape == (batch_size,)
    print(f"  ✓ All {3} calls succeeded")

    # Test 4: Different batch sizes
    print("\n[Test 4] Different batch sizes...")
    for test_batch in [1, 2, 5]:
        sample_key = jax.random.PRNGKey(20 + test_batch)
        actions_test, logp_test = jitted_fn(params, sample_key, test_batch)
        assert actions_test.shape == (test_batch, num_agents)
        assert logp_test.shape == (test_batch,)
        print(f"  ✓ Batch size {test_batch} works")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("The AutoregressiveTabularPolicy works correctly in jitted context.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
