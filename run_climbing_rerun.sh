#!/usr/bin/env bash
#
# Climbing rerun on the FIXED matrax environment — single GPU.
#
# Reproduces the `fixed-matrix-games-climbing` experiment exactly: same CARBS-selected
# hyperparameters per algorithm, same 10 seeds, same 19,988,480 environment steps,
# same 122 evaluations of 32 episodes.
#
# Ten algorithms. Seven carry the original TPU-sweep hyperparameters from
# optimal_parameters/matrix-games-carbs-sweep.csv; three tabular variants
# (factored / autoregressive / chained autoregressive) were tuned later by
# tune_climbing_missing.py under the same CARBS protocol -- 40 trials on this same
# fixed Climbing game -- and their values come from tuning_results/climbing-carbs-best.csv.
#
# Budget note for the three late additions
#   CARBS tuned their num_updates to ~1190 (~9.7M steps); the original seven all tuned
#   to the 2440 maximum. This script runs every algorithm at 2440 / 19,988,480 anyway:
#   marl-eval aligns algorithms by evaluation index, so all ten must produce the same
#   122 points or the curves cannot be aggregated onto shared axes. Their other tuned
#   hyperparameters are used verbatim; only the horizon is made uniform, and running
#   past a converged tabular policy does not change where it converged.
#
# Running a subset
#   The marl-eval JSON logger MERGES into an existing metrics.json, so to add just the
#   three without retouching the seven:
#     ./run_climbing_rerun.sh ff_ppo_central_factored_tabular \
#         ff_ppo_central_autoreg_tabular ff_ppo_central_autoreg_chained_tabular
#
# Device note
#   total_timesteps = n_devices x num_updates x rollout_length x update_batch_size x num_envs
#   The original ran on a TPU-v4 with 4 devices at update_batch_size=2, num_envs=8,
#   i.e. 4 x 2 x 8 = 64 parallel environment copies. Gradients are pmean'd over BOTH the
#   `batch` (update_batch_size) and `device` axes and the learner is broadcast across the
#   batch axis, so devices and update_batch_size are mathematically interchangeable.
#   On one GPU we therefore use update_batch_size=8, num_envs=8 -> 1 x 8 x 8 = 64.
#   This is an exact reproduction, not an approximation. The script computes this and
#   refuses to run if it cannot hit 64.
#
# Output
#   ONE json file, marl-eval schema, at:
#     results/json/climbing-rerun-fixed-matrax/metrics.json
#   No wandb, no Neptune.
#
# Usage
#   chmod +x run_climbing_rerun.sh
#   ./run_climbing_rerun.sh                 # all 7 algorithms
#   ./run_climbing_rerun.sh ff_ippo ff_sable   # a subset
#
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
ENVCFG=matrax                              # scenario.task_name defaults to Climbing-stateless-v0
TOTAL_TIMESTEPS=19988480
ROLLOUT=128
NUM_ENVS=8
TARGET_COPIES=64                           # n_devices x update_batch_size x num_envs
SEEDS="0,1,2,3,4,5,6,7,8,9"
JSON_TAG=climbing-rerun-fixed-matrax

BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'

# ---------------------------------------------------------------- preflight
echo "${BOLD}== Preflight ==${OFF}"
UBS=$($PY - "$TARGET_COPIES" "$NUM_ENVS" <<'PYEOF'
import sys
try:
    import jax
    n = len(jax.devices())
    kinds = {d.platform for d in jax.devices()}
except Exception as e:
    print(f"ERR jax import failed: {e}"); sys.exit(0)
target, num_envs = int(sys.argv[1]), int(sys.argv[2])
if target % (n * num_envs) != 0:
    print(f"ERR {target} parallel copies is not divisible by n_devices({n}) x num_envs({num_envs}). "
          f"Pin to one GPU with CUDA_VISIBLE_DEVICES=0, or adjust NUM_ENVS.")
    sys.exit(0)
print(f"OK {target // (n * num_envs)} {n} {','.join(sorted(kinds))}")
PYEOF
)
case "$UBS" in
  OK*) read -r _ UPDATE_BATCH NDEV PLATFORM <<<"$UBS"
       echo "  ${GRN}✓${OFF} devices: $NDEV ($PLATFORM)"
       echo "  ${GRN}✓${OFF} update_batch_size = $UPDATE_BATCH  (num_envs=$NUM_ENVS -> $((NDEV*UPDATE_BATCH*NUM_ENVS)) parallel copies)" ;;
  *)   echo "  ${RED}✗${OFF} ${UBS#ERR }"; exit 1 ;;
esac

# --- matrax must contain the cooperative-climbing fix (commit c2b92bb) -------
MX=$($PY - <<'PYEOF'
try:
    import jax.numpy as jnp
    from matrax.games.climbing import climbing_game as g
    import numpy as np
    a = np.asarray(g)
    if a.shape[0] != 2:
        print("ERR unexpected shape " + str(a.shape)); raise SystemExit
    same = bool((a[0] == a[1]).all())
    print(("ERR both agents share one payoff matrix -> PRE-FIX matrax"
           if same else "OK agent 2 sees the transpose -> fixed matrax"))
except Exception as e:
    print(f"ERR could not import matrax: {e}")
PYEOF
)
case "$MX" in
  OK*) echo "  ${GRN}✓${OFF} matrax: ${MX#OK }" ;;
  *)   echo "  ${RED}✗${OFF} matrax: ${MX#ERR }"
       echo "       fix with:"
       echo "         pip install --force-reinstall --no-deps --no-cache-dir \\"
       echo "           'matrax @ git+https://github.com/RuanJohn/matrax@c2b92bb'"
       exit 1 ;;
esac

UPDATES=$(( TOTAL_TIMESTEPS / ROLLOUT / UPDATE_BATCH / NUM_ENVS / NDEV ))
echo "  ${GRN}✓${OFF} derived num_updates = $UPDATES  (target 2440)"
[ "$UPDATES" -ne 2440 ] && { echo "  ${RED}✗${OFF} num_updates != 2440 — stopping."; exit 1; }
echo "  ${GRN}✓${OFF} output -> results/json/${JSON_TAG}/metrics.json"
echo

# ---------------------------------------------------------------- shared overrides
SHARED=(
  "env=${ENVCFG}"
  "system.total_timesteps=${TOTAL_TIMESTEPS}"
  "system.rollout_length=${ROLLOUT}"
  "system.update_batch_size=${UPDATE_BATCH}"
  "system.decay_learning_rates=False"
  "arch.num_envs=${NUM_ENVS}"
  "arch.num_evaluation=122"
  "arch.num_eval_episodes=32"
  "arch.absolute_metric=True"
  "arch.num_absolute_metric_eval_episodes=320"
  "arch.evaluation_greedy=False"
  "logger.use_wandb=False"
  "logger.use_neptune=False"
  "logger.use_tb=False"
  "logger.use_json=True"
  "logger.use_console=True"
  "logger.kwargs.upload_json_data=False"
  "logger.kwargs.json_path=${JSON_TAG}"
)

# ------------------------------------------- CARBS-selected, per algorithm
# Taken verbatim from wandb tag `fixed-matrix-games-climbing` (the tuned final runs).
hparams () {
  case "$1" in
    ff_ippo) echo \
      "system.actor_lr=7.219151825657473e-05 system.critic_lr=0.0001438574247304 \
       system.ent_coef=0.0100075046134918 system.clip_eps=0.1908758634086376 \
       system.gae_lambda=0.8543278123034208 system.gamma=0.9428826104701992 \
       system.max_grad_norm=1.2840210846031126 system.num_minibatches=8 \
       system.ppo_epochs=3 system.vf_coef=0.2569252853677521 system.add_agent_id=True" ;;
    ff_mappo) echo \
      "system.actor_lr=0.0001863675281119 system.critic_lr=9.039766392413592e-05 \
       system.ent_coef=0.008119222203987 system.clip_eps=0.3004362058320515 \
       system.gae_lambda=0.8102979571402557 system.gamma=0.9648534276738628 \
       system.max_grad_norm=1.432324869821879 system.num_minibatches=8 \
       system.ppo_epochs=8 system.vf_coef=0.5865513205945188 system.add_agent_id=True" ;;
    ff_ppo_central) echo \
      "system.actor_lr=0.0003625444151683 system.critic_lr=2.4125314965766908e-05 \
       system.ent_coef=0.0196168325215803 system.clip_eps=0.0103378510425279 \
       system.gae_lambda=0.8658675877868366 system.gamma=0.3615259794286558 \
       system.max_grad_norm=2.552865475353602 system.num_minibatches=4 \
       system.ppo_epochs=9 system.vf_coef=0.0395216779431527 system.add_agent_id=False" ;;
    ff_ppo_central_factored) echo \
      "system.actor_lr=0.0007354502612595 system.critic_lr=0.0002431419896703 \
       system.ent_coef=0.0003663501870516 system.clip_eps=0.0910527782461841 \
       system.gae_lambda=0.9567493558250868 system.gamma=0.1188092997146624 \
       system.max_grad_norm=0.4642040087804637 system.num_minibatches=8 \
       system.ppo_epochs=6 system.vf_coef=0.0128038955892509 system.add_agent_id=False" ;;
    ff_sable) echo \
      "system.actor_lr=0.0002168771647924 \
       system.ent_coef=0.0071303373064948 system.clip_eps=0.2143806939671704 \
       system.gae_lambda=0.9954439410325642 system.gamma=0.9995625708010296 \
       system.max_grad_norm=0.1186226302727092 system.num_minibatches=8 \
       system.ppo_epochs=6 system.vf_coef=0.1013847196570545 system.add_agent_id=True" ;;
    ff_ippo_tabular_split) echo \
      "system.actor_lr=0.0005952745713025 system.critic_lr=0.0003878721629335 \
       system.ent_coef=9.853680153679056e-05 system.clip_eps=0.0206710378703315 \
       system.gae_lambda=0.9089191164903734 system.gamma=0.380582366448322 \
       system.max_grad_norm=0.3196764724828702 system.num_minibatches=8 \
       system.ppo_epochs=9 system.vf_coef=0.0120200585417353 system.add_agent_id=True" ;;
    ff_ppo_central_tabular) echo \
      "system.actor_lr=0.0002221821941983 system.critic_lr=3.584704598404175e-05 \
       system.ent_coef=0.0071468320905153 system.clip_eps=0.0103743325850989 \
       system.gae_lambda=0.7375058453984705 system.gamma=0.4659544125359651 \
       system.max_grad_norm=4.307290361442906 system.num_minibatches=8 \
       system.ppo_epochs=8 system.vf_coef=0.0666047805329527 system.add_agent_id=False" ;;
    # --- tuned 2026-09-25 by tune_climbing_missing.py (CARBS, 40 trials each,
    #     Climbing-stateless-v0, post-fix matrax). Source:
    #     tuning_results/climbing-carbs-best.csv
    ff_ppo_central_factored_tabular) echo \
      "system.actor_lr=0.00011112021076972583 system.critic_lr=0.00016468383156859992 \
       system.ent_coef=0.009706841795231276 system.clip_eps=0.0847173821830581 \
       system.gae_lambda=0.7534770546856423 system.gamma=0.9014857509700724 \
       system.max_grad_norm=5.775572112893504 system.num_minibatches=8 \
       system.ppo_epochs=3 system.vf_coef=0.18607979369617303 system.add_agent_id=False" ;;
    ff_ppo_central_autoreg_tabular) echo \
      "system.actor_lr=0.00010643760149201132 system.critic_lr=0.00012511986833019535 \
       system.ent_coef=0.011183353770572585 system.clip_eps=0.05406721950724329 \
       system.gae_lambda=0.6310405737267282 system.gamma=0.9473787126135559 \
       system.max_grad_norm=2.8215254880664085 system.num_minibatches=8 \
       system.ppo_epochs=5 system.vf_coef=0.05499893748883632 system.add_agent_id=False" ;;
    ff_ppo_central_autoreg_chained_tabular) echo \
      "system.actor_lr=0.00018066075983041944 system.critic_lr=0.0004176189812072225 \
       system.ent_coef=0.00724932682823488 system.clip_eps=0.06429642407568345 \
       system.gae_lambda=0.323195639497718 system.gamma=0.6193037975192678 \
       system.max_grad_norm=1.2592191563515756 system.num_minibatches=8 \
       system.ppo_epochs=3 system.vf_coef=0.34394618760624096 system.add_agent_id=False" ;;
    *) echo ""; return 1 ;;
  esac
}

entrypoint () {
  case "$1" in
    ff_sable) echo "mava/systems/sable/anakin/ff_sable.py" ;;
    *)        echo "mava/systems/ppo/anakin/$1.py" ;;
  esac
}

ALGOS=(ff_ippo ff_mappo ff_ppo_central ff_ppo_central_factored ff_sable
       ff_ippo_tabular_split ff_ppo_central_tabular
       ff_ppo_central_factored_tabular ff_ppo_central_autoreg_tabular
       ff_ppo_central_autoreg_chained_tabular)
[ $# -gt 0 ] && ALGOS=("$@")

# ---------------------------------------------------------------- run
FAILED=()
START=$(date +%s)
for algo in "${ALGOS[@]}"; do
  HP=$(hparams "$algo") || { echo "${RED}unknown algorithm: $algo${OFF}"; FAILED+=("$algo"); continue; }
  SCRIPT=$(entrypoint "$algo")
  [ -f "$SCRIPT" ] || { echo "${RED}missing entrypoint: $SCRIPT${OFF}"; FAILED+=("$algo"); continue; }
  echo "${BOLD}== $algo ==${OFF}  ($SCRIPT, 10 seeds)"
  # shellcheck disable=SC2086
  $PY "$SCRIPT" -m "${SHARED[@]}" $HP "system.seed=${SEEDS}"
  if [ $? -ne 0 ]; then echo "  ${RED}✗ $algo failed${OFF}"; FAILED+=("$algo"); else echo "  ${GRN}✓ $algo done${OFF}"; fi
  echo
done
END=$(date +%s)
echo "${BOLD}== Finished in $(( (END-START)/60 ))m $(( (END-START)%60 ))s ==${OFF}"
[ ${#FAILED[@]} -gt 0 ] && echo "  ${RED}failed: ${FAILED[*]}${OFF}"

# ---------------------------------------------------------------- verify
JSON="results/json/${JSON_TAG}/metrics.json"
echo
echo "${BOLD}== Result ==${OFF}"
if [ ! -f "$JSON" ]; then
  echo "  ${RED}✗ $JSON not found${OFF}"; exit 1
fi
echo "  file: $JSON  ($(du -h "$JSON" | cut -f1))"
$PY - "$JSON" <<'PYEOF'
import json, sys, statistics as st
d = json.load(open(sys.argv[1]))
print("\n  Climbing payoff: [[11,-30,0],[-30,7,?],[?,?,5]]   optimum 11, shadowed equilibrium 5\n")
print(f"  {'algorithm':30s}{'seeds':>7s}{'final return':>16s}{'distinct finals':>34s}")
for env, tasks in d.items():
    for task, algos in tasks.items():
        for algo in sorted(algos):
            runs = algos[algo]
            finals = []
            for s, steps in runs.items():
                am = steps.get("absolute_metrics")
                if am is None:
                    ks = sorted((k for k in steps if k.startswith("step_")),
                                key=lambda k: int(k.split("_")[1]))
                    if not ks: continue
                    am = steps[ks[-1]]
                v = am.get("mean_episode_return")
                if isinstance(v, list): v = sum(v)/len(v)
                if v is not None: finals.append(float(v))
            if not finals: continue
            uniq = sorted({round(f, 2) for f in finals})
            print(f"  {algo:30s}{len(finals):>7d}{st.mean(finals):>16.2f}   {uniq}")
print("\n  Expected if the fix behaves as the post-fix runs suggest:")
print("    5.00 -> ff_ippo, ff_mappo, ff_ppo_central_factored, ff_ippo_tabular_split")
print("   11.00 -> ff_ppo_central, ff_sable, ff_ppo_central_tabular")
PYEOF
echo
echo "  Send me this one file:  $JSON"
