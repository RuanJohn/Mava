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
Test script to verify AutoregressiveTabularPolicy.train_fn works correctly.

Run with: source .venv/bin/activate && python test_train_fn.py
"""

import jax
import jax.numpy as jnp

from mava.networks.heads import AutoregressiveTabularPolicy


def main():
    print("=" * 60)
    print("Testing AutoregressiveTabularPolicy.train_fn")
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

    # Define apply functions
    def act_fn(params, sample_key, batch_size):
        return network.apply(params, sample_key, batch_size, method="act_fn")

    def train_fn(params, sample_key, batch_size, action):
        return network.apply(params, sample_key, batch_size, action, method="train_fn")

    # Test 1: Generate actions and compute their log probs
    print("\n[Test 1] Generate actions and compute log probs (non-jitted)...")
    sample_key = jax.random.PRNGKey(1)
    actions, logp_act = act_fn(params, sample_key, batch_size)
    assert actions.shape == (
        batch_size,
        num_agents,
    ), f"Expected actions shape ({batch_size}, {num_agents}), got {actions.shape}"
    assert logp_act.shape == (
        batch_size,
    ), f"Expected logp shape ({batch_size},), got {logp_act.shape}"
    print(f"  ✓ Generated actions shape: {actions.shape}")
    print(f"  ✓ Logp from act_fn shape: {logp_act.shape}")

    # Compute log probs using train_fn
    sample_key_train = jax.random.PRNGKey(2)
    logp_train = train_fn(params, sample_key_train, batch_size, actions)
    assert logp_train.shape == (
        batch_size,
    ), f"Expected logp shape ({batch_size},), got {logp_train.shape}"
    print(f"  ✓ Logp from train_fn shape: {logp_train.shape}")

    # Verify that log probs match (they should be the same for the same actions)
    # Note: They might differ slightly due to numerical precision, but should be very close
    max_diff = jnp.max(jnp.abs(logp_act - logp_train))
    print(f"  ✓ Max difference between act_fn and train_fn logp: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Log probs should match! Max diff: {max_diff}"

    # Test 2: Jitted train_fn
    print("\n[Test 2] Jitted train_fn execution...")
    jitted_train_fn = jax.jit(train_fn, static_argnums=2)

    sample_key = jax.random.PRNGKey(3)
    actions, _ = act_fn(params, sample_key, batch_size)
    sample_key_train = jax.random.PRNGKey(4)
    logp_train_jit = jitted_train_fn(params, sample_key_train, batch_size, actions)

    assert logp_train_jit.shape == (
        batch_size,
    ), f"Expected logp shape ({batch_size},), got {logp_train_jit.shape}"
    print(f"  ✓ Jitted train_fn logp shape: {logp_train_jit.shape}")

    # Verify it matches non-jitted version
    logp_train_nonjit = train_fn(params, sample_key_train, batch_size, actions)
    max_diff = jnp.max(jnp.abs(logp_train_jit - logp_train_nonjit))
    print(f"  ✓ Max difference between jitted and non-jitted: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Jitted and non-jitted should match! Max diff: {max_diff}"

    # Test 3: Different batch sizes
    print("\n[Test 3] Different batch sizes...")
    for test_batch in [1, 2, 5]:
        sample_key = jax.random.PRNGKey(10 + test_batch)
        actions_test, _ = act_fn(params, sample_key, test_batch)
        sample_key_train = jax.random.PRNGKey(20 + test_batch)
        logp_test = jitted_train_fn(params, sample_key_train, test_batch, actions_test)
        assert logp_test.shape == (
            test_batch,
        ), f"Expected logp shape ({test_batch},), got {logp_test.shape}"
        print(f"  ✓ Batch size {test_batch} works (logp shape: {logp_test.shape})")

    # Test 4: Verify log probs are reasonable (should be negative)
    print("\n[Test 4] Verify log probabilities are reasonable...")
    sample_key = jax.random.PRNGKey(30)
    actions_test, _ = act_fn(params, sample_key, batch_size)
    sample_key_train = jax.random.PRNGKey(31)
    logp_test = train_fn(params, sample_key_train, batch_size, actions_test)
    assert jnp.all(logp_test < 0), "Log probabilities should be negative"
    assert jnp.all(jnp.isfinite(logp_test)), "Log probabilities should be finite"
    print("  ✓ All log probs are negative and finite")
    print(f"  ✓ Log prob range: [{jnp.min(logp_test):.2f}, {jnp.max(logp_test):.2f}]")

    # Test 5: Verify consistency - same actions should give same log probs
    print("\n[Test 5] Verify consistency with same actions...")
    sample_key = jax.random.PRNGKey(40)
    actions_test, _ = act_fn(params, sample_key, batch_size)
    sample_key_train1 = jax.random.PRNGKey(41)
    sample_key_train2 = jax.random.PRNGKey(42)
    logp1 = train_fn(params, sample_key_train1, batch_size, actions_test)
    logp2 = train_fn(params, sample_key_train2, batch_size, actions_test)
    max_diff = jnp.max(jnp.abs(logp1 - logp2))
    print(f"  ✓ Max difference with different keys (should be 0): {max_diff:.2e}")
    assert max_diff < 1e-5, "Same actions should give same log probs regardless of key"

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("The AutoregressiveTabularPolicy.train_fn works correctly.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
