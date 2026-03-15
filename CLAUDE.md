# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Mava

Mava is a distributed multi-agent reinforcement learning (MARL) framework built entirely in JAX. It follows the **PureJaxRL/CleanRL philosophy**: single-file implementations per algorithm for research clarity, not a modular library. The design prioritizes experimentation over abstraction.

## Common Commands

```bash
# Install
pip install -e .

# Run a system (defaults to Robotic Warehouse, 2 agents)
python mava/systems/ppo/anakin/ff_ippo.py

# Run with different environment or config overrides (Hydra)
python mava/systems/ppo/anakin/ff_ippo.py env=lbf
python mava/systems/ppo/anakin/ff_ippo.py system.num_updates=1000 arch.num_envs=16

# Lint (ruff, line-length 100)
ruff check mava/ --fix
ruff format mava/

# Type check
python -m mypy mava/

# Run all tests
pytest test/

# Run a single test
pytest test/integration_test.py -k "test_ff_ippo"

# Run pre-commit hooks
pre-commit run --all-files
```

## Architecture Overview

### Core Abstractions (`mava/types.py`)

- **`MarlEnv` Protocol** — standard environment interface (`reset`, `step`, `num_agents`, `time_limit`, `action_dim`)
- **`LearnerState`** — NamedTuple carrying all training state through JAX `scan` loops
- **`ExperimentOutput[MavaState]`** — wraps learner_state + episode_metrics + train_metrics
- **`LearnerFn`** — type alias for `(LearnerState, Any) -> (LearnerState, Metrics)`, returned by `get_learner_fn()`

### System Structure (`mava/systems/`)

Each algorithm lives in a self-contained file. Entry points:
- `run_experiment(config: DictConfig) -> float` — runs training, returns performance metric
- `get_learner_fn(env, apply_fns, update_fns, config) -> LearnerFn` — builds the `jax.lax.scan`-compatible update step

Systems are organized as:
```
systems/
  ppo/anakin/   # ff_ippo, ff_mappo, rec_ippo, rec_mappo, tabular/autoregressive variants
  ppo/sebulba/  # Alternative multi-device backend
  sac/          # ff_isac, ff_masac, ff_hasac
  q_learning/   # rec_iql, rec_qmix
  mat/          # Multi-Agent Transformer
  sable/        # Off-policy MARL
systems_tuning/ # Hyperparameter-tuned copies of above systems
```

**CTDE vs DTDE**: `mappo`/`masac` use centralized training (global state for critic), `ippo`/`isac` use decentralized (each agent sees only local obs).

### Configuration (`mava/configs/`)

Hydra-based structured configs. Each system's `default/*.yaml` composes from:
- `arch/` — `anakin` (single-device vmap) or `sebulba` (multi-device pmap)
- `system/` — algorithm hyperparameters (lr, rollout_len, clip_eps, etc.)
- `network/` — `mlp`, `cnn`, `rnn`, `transformer`, `retention`
- `env/` — environment and scenario selection
- `logger/` — Neptune / WandB / JSON / TensorBoard

### Networks (`mava/networks/`)

- `base.py` — `FeedForwardActor`, `FeedForwardValueNet`, `RecurrentActor`, `RecurrentValueNet`, Q-network variants
- `heads.py` — `CategoricalHead`, `GaussianHead`, `MaskedCategoricalHead`, autoregressive variants
- `distributions.py` — policy distribution wrappers
- `attention.py`, `retention.py` — Transformer/ALiBi components for MAT/SABLE

### Environment Wrappers (`mava/wrappers/`)

All wrappers conform to `MarlEnv` protocol. Key ones:
- `jumanji.py` — RobotWarehouse, LevelBasedForaging, Connector, Cleaner
- `jaxmarl.py` — SMAX, MaBrax, MPE (wraps JaxMARL envs)
- `gym.py` — `GymToJumanji` for Gymnasium compatibility
- `episode_metrics.py` — injects episode return/length/win-rate into `TimeStep.extras`
- `centralised_controller.py` — flattens multi-agent obs/actions for centralized critic

### Logging (`mava/utils/logger.py`)

`MavaLogger` supports: Neptune, WandB, TensorBoard, JSON (MARL-eval format). Logger backend configured via `logger/` config group.

### Checkpointing (`mava/utils/checkpointing.py`)

Orbax-based model saving/loading. Configured via `arch.checkpoint` in configs.

## Key JAX Patterns

1. **No mutable state** — all state is passed explicitly as NamedTuples through `jax.lax.scan`
2. **Parallelization** — `vmap` over environments within a device; `pmap` across devices (sebulba)
3. **RNG handling** — keys are split and passed explicitly; never use global RNG
4. **JIT boundaries** — `get_learner_fn` returns a function that gets `jax.jit`-compiled (or `pmap`-compiled)

## Testing

`test/integration_test.py` has parametrized tests for all systems × environments using a `fast_config` fixture (1-2 training steps, batch size 1). Tests verify a forward pass completes without error, not convergence.
