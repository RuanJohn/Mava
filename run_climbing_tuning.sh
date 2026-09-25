#!/usr/bin/env bash
#
# CARBS sweep on the fixed Climbing game for the three tabular policies that never
# had one, so the full factorisation ladder can be reported at its own tuned
# configuration rather than only under shared hyperparameters.
#
#   ./run_climbing_tuning.sh                                    # all three
#   ./run_climbing_tuning.sh ff_ppo_central_autoreg_tabular     # just one
#
# Same protocol as the original seven: CARBS, 40 trials per algorithm, 11-parameter
# space from mava/systems_tuning/tune_space.py, objective = final evaluator return,
# cost = num_updates.
#
# Budget: total = n_devices x num_updates x rollout x update_batch_size x num_envs.
#   CARBS's own maximum suggestion (2440 updates) is exactly 19,988,480 steps -- the
#   run_climbing_rerun.sh budget. The 20M cap is arithmetic, not a clamp; trials span
#   1.31M..19.99M and cannot exceed it.
#
# update_batch_size on one GPU
#   Gradients are pmean'd over BOTH the `device` and `batch` (update_batch_size) axes,
#   so only their product matters. The original tuning ran on a TPU-v4 at 4 devices x
#   ubs 2 x 8 envs = 8 learner copies, batch 1024 each. On ONE GPU the identical
#   configuration is ubs=8 (1 x 8 x 8 = 64 copies, still 8 learner copies, still 1024
#   per copy). ubs=8 here is not a bigger batch -- it is the same batch on one device,
#   and it is what run_climbing_rerun.sh already used to produce the thesis data.
#   The preflight checks learner copies == 8, not ubs, so it adapts to any device count.
#
# Pinned to one device by default; JAX would otherwise shard across every visible
# device and silently change the configuration.
#
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; OFF=$'\033[0m'

PY=""
for c in "./.venv/bin/python" "$(command -v python3 || true)"; do
  [ -x "$c" ] || continue
  if "$c" -c "import jax, carbs, mava, matrax" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  printf '  %s✗%s no interpreter with jax + carbs + mava + matrax\n' "$RED" "$OFF"
  echo ""
  echo "      Same recipe as the original TPU sweep (make_tpus/carbs/Makefile:113-120),"
  echo "      with the TPU jax swapped for the CUDA build:"
  echo ""
  echo "        pip install -e ."
  echo "        pip install torch torchvision torchaudio \\"
  echo "            --index-url https://download.pytorch.org/whl/cpu   # CARBS needs torch;"
  echo "                                                               # CPU build on purpose,"
  echo "                                                               # so it does not take"
  echo "                                                               # GPU memory from JAX"
  echo "        pip install git+https://github.com/imbue-ai/carbs"
  echo "        pip install \"jax[cuda12]==0.4.30\""
  echo "        pip install --force-reinstall --no-deps --no-cache-dir \\"
  echo "            \"matrax @ git+https://github.com/RuanJohn/matrax@c2b92bb\""
  exit 1
fi
printf '%s== Climbing CARBS sweep ==%s\n  %s✓%s interpreter: %s\n' "$BOLD" "$OFF" "$GRN" "$OFF" "$PY"

# ~120 trials total (3 x 40); each is 1.3M-20M steps on a 2-agent 3-action game.
"$PY" tune_climbing_missing.py "$@"
RC=$?

if [ "$RC" -eq 0 ]; then
  printf '\n  %s✓%s done. Next:\n' "$GRN" "$OFF"
  echo "     1. inspect tuning_results/climbing-carbs-*-trials.csv"
  echo "     2. append tuning_results/climbing-carbs-best.csv rows to"
  echo "        optimal_parameters/matrix-games-carbs-sweep.csv"
  echo "     3. add the three algorithms to run_climbing_rerun.sh and rerun 10 seeds"
else
  printf '\n  %s✗%s sweep exited %s\n' "$RED" "$OFF" "$RC"
fi
exit $RC
