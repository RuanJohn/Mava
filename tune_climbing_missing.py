"""CARBS sweep on the fixed Climbing game for the three policies that never got one.

Why this exists
---------------
optimal_parameters/matrix-games-carbs-sweep.csv holds per-(algorithm, task) tuned
hyperparameters for seven policies, including a row for Climbing-stateless-v0 itself.
Three tabular variants have no row, which is why they appear only in the
shared-hyperparameter ladder and were left out of the tuned rerun:

    ff_ppo_central_factored_tabular          N*A            ladder result 5.00
    ff_ppo_central_autoreg_tabular           sum_i A^(i+1)  ladder result 11.00
    ff_ppo_central_autoreg_chained_tabular   A+(N-1)A^2     ladder result 11.00

This runs the same protocol for those three so the whole ladder can be reported at
its own tuned configuration.

Protocol -- identical to the original sweep
-------------------------------------------
* CARBS (cost-aware Pareto-region Bayesian search), a fresh instance per algorithm
* 40 trials each
* the 11-parameter space in mava/systems_tuning/tune_space.py, with the Sable-only
  `decay_kappa` removed exactly as the original scripts do
* objective  = final evaluator mean episode return (maximised)
* cost       = num_updates, the term that makes the search cost-aware
* num_updates = 10 x suggestion, i.e. 160 .. 2440 updates

Budget
------
total_timesteps = n_devices x num_updates x rollout_length x update_batch_size x num_envs

With rollout_length=128, num_envs=8 and n_devices x update_batch_size = 8, CARBS's own
maximum (2440 updates) lands on 19,988,480 steps -- the same 20M budget as
run_climbing_rerun.sh. The cap is a property of the arithmetic, not a clamp, so the
search explores 1.31M .. 19.99M steps and can never exceed the rerun budget.

Unlike the original scripts this one persists every trial to CSV: Neptune is
discontinued, and that was previously the only durable record of a sweep.

    ./run_climbing_tuning.sh              # all three
    ./run_climbing_tuning.sh ff_ppo_central_autoreg_tabular
"""

import argparse
import copy
import csv
import importlib
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "mava", "configs", "default")
OUT_DIR = os.path.join(ROOT, "tuning_results")

ALGOS = [
    "ff_ppo_central_factored_tabular",
    "ff_ppo_central_autoreg_tabular",
    "ff_ppo_central_autoreg_chained_tabular",
]

TASK, ENV_NAME, ENV_CFG = "Climbing-stateless-v0", "matrax", "matrax"

# Identical to run_climbing_rerun.sh -- same constants, same arithmetic, same guard.
#   total = n_devices x num_updates x rollout_length x update_batch_size x num_envs
# Gradients are pmean'd over BOTH the `device` and `batch` (update_batch_size) axes and
# the learner is broadcast across the batch axis, so only their PRODUCT matters. The
# original tuning ran on a TPU-v4: 4 devices x ubs 2 x 8 envs = 64 parallel copies,
# 8 learner copies, batch 128x8=1024 per copy. On one GPU, ubs=8 reproduces that
# exactly (1 x 8 x 8 = 64, still 8 learner copies, still 1024 per copy) -- it is the
# same configuration expressed on different hardware, not a larger batch.
ROLLOUT, NUM_ENVS = 128, 8
TARGET_COPIES = 64          # n_devices x update_batch_size x num_envs (as in the rerun)
REF_LEARNER_COPIES = 8      # TPU-v4 reference: 4 devices x ubs 2
N_TRIALS = 40
MAX_UPDATES = 2440          # CARBS max suggestion (244) x 10
TOTAL_AT_MAX = 19_988_480   # the run_climbing_rerun.sh budget

# Columns of optimal_parameters/matrix-games-carbs-sweep.csv, so the best rows can be
# concatenated straight onto the existing table.
SWEEP_COLS = ["system_name", "task", "env_name", "run_id", "episode_return", "win_rate",
              "time_series_total", "wall_clock_time", "num_minibatches", "vf_coef",
              "max_grad_norm", "decay_scaling_factor", "ent_coef", "actor_lr",
              "gae_lambda", "critic_lr", "gamma", "ppo_epochs", "clip_eps", "num_envs",
              "num_updates", "num_evaluation"]


def preflight():
    """Refuse to run against a pre-fix matrax, and work out update_batch_size."""
    import jax
    import matrax.games.climbing as cg

    a = np.asarray(cg.climbing_game)
    if a.shape[0] < 2 or np.array_equal(a[0], a[1]):
        sys.exit("matrax is PRE-FIX (both agents see the same payoff matrix).\n"
                 "  Reinstall:  pip install --force-reinstall --no-deps --no-cache-dir "
                 '"matrax @ git+https://github.com/RuanJohn/matrax@c2b92bb"')
    print(f"  matrax FIXED   agent 1 {a[0].tolist()}\n"
          f"                 agent 2 {a[1].tolist()}")

    n_dev = len(jax.devices())
    plat = jax.devices()[0].platform
    if TARGET_COPIES % (n_dev * NUM_ENVS):
        sys.exit(f"{TARGET_COPIES} parallel copies is not divisible by "
                 f"n_devices({n_dev}) x num_envs({NUM_ENVS}). Pin to one GPU with "
                 "CUDA_VISIBLE_DEVICES=0, or adjust NUM_ENVS.")
    ubs = TARGET_COPIES // (n_dev * NUM_ENVS)
    copies = n_dev * ubs
    total_max = n_dev * MAX_UPDATES * ROLLOUT * ubs * NUM_ENVS
    print(f"  devices: {n_dev} ({plat})")
    print(f"  update_batch_size = {ubs}  (num_envs={NUM_ENVS} -> "
          f"{n_dev * ubs * NUM_ENVS} parallel copies)")
    print(f"  learner copies = n_devices x ubs = {copies}   "
          f"batch per copy = {ROLLOUT * NUM_ENVS}")

    # The invariant that must hold, not the value of ubs on its own: ubs trades off
    # against device count, so on one GPU it is necessarily 8.
    if copies != REF_LEARNER_COPIES:
        sys.exit(f"learner copies {copies} != {REF_LEARNER_COPIES} (TPU-v4 reference: "
                 f"4 devices x ubs 2). This would not match how the other seven were "
                 f"tuned - refusing to run.")
    print(f"  matches TPU-v4 reference (4 devices x ubs 2 x {NUM_ENVS} envs)")

    derived = total_max // ROLLOUT // ubs // NUM_ENVS // n_dev
    if derived != MAX_UPDATES or total_max != TOTAL_AT_MAX:
        sys.exit(f"derived num_updates {derived} != {MAX_UPDATES} or budget "
                 f"{total_max:,} != {TOTAL_AT_MAX:,} - refusing to run off-budget.")
    print(f"  budget:  max trial {MAX_UPDATES} updates -> {total_max:,} steps "
          f"(= run_climbing_rerun.sh)")
    return ubs, n_dev


def build_cfg(algo, ubs):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.2"):
        cfg = compose(config_name=f"{algo}.yaml", overrides=[
            f"env={ENV_CFG}",
            f"env.scenario.task_name={TASK}",
            f"arch.num_envs={NUM_ENVS}",
            f"system.rollout_length={ROLLOUT}",
            f"system.update_batch_size={ubs}",
            "logger.use_wandb=False",
            "logger.use_neptune=False",
            "logger.use_json=False",
            "arch.absolute_metric=False",   # objective is the final eval, as in the original
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.system.total_timesteps = None       # let num_updates drive the budget
    return cfg


def sweep(algo, ubs, n_dev, rng):
    from carbs import CARBS, ObservationInParam
    from mava.systems_tuning.tune_space import carbs_params, param_spaces

    mod = importlib.import_module(f"mava.systems_tuning.ppo.anakin.{algo}")
    run_experiment = mod.run_experiment

    # Copy, never mutate: the originals do `del param_spaces[-1]`, which would corrupt
    # the shared module-level list on the second and third algorithm in one process.
    spaces = [p for p in param_spaces if p.name != "decay_kappa"]
    assert len(spaces) == 11, f"expected 11 params, got {len(spaces)}"

    base = build_cfg(algo, ubs)
    carbs = CARBS(carbs_params, spaces)

    os.makedirs(OUT_DIR, exist_ok=True)
    trials_path = os.path.join(OUT_DIR, f"climbing-carbs-{algo}-trials.csv")
    tf = open(trials_path, "w", newline="")
    tw = csv.writer(tf)
    tw.writerow(["trial", "status", "episode_return", "wall_clock_s", "num_updates",
                 "total_timesteps", "seed"] + [p.name for p in spaces])
    tf.flush()

    print(f"\n=== {algo} : {N_TRIALS} CARBS trials ===")
    best, best_sug, ok, fail = -np.inf, None, 0, 0

    for t in range(1, N_TRIALS + 1):
        sug = carbs.suggest().suggestion
        cfg = copy.deepcopy(base)
        seed = int(rng.integers(1, 1_000_000))
        cfg.system.seed = seed
        cfg.system.actor_lr = sug["actor_lr"]
        cfg.system.critic_lr = sug["critic_lr"]
        cfg.system.ppo_epochs = sug["ppo_epochs"]
        cfg.system.num_minibatches = int(2 ** sug["num_minibatches"])
        cfg.system.gamma = sug["gamma"]
        cfg.system.gae_lambda = sug["gae_lambda"]
        cfg.system.clip_eps = sug["clip_eps"]
        cfg.system.ent_coef = sug["ent_coef"]
        cfg.system.vf_coef = sug["vf_coef"]
        cfg.system.max_grad_norm = sug["max_grad_norm"]
        n_upd = int(sug["num_updates"] * 10)
        cfg.system.num_updates = n_upd
        cfg.arch.num_evaluation = int(sug["num_updates"])
        total = n_dev * n_upd * ROLLOUT * ubs * NUM_ENVS

        row = [t, "", "", "", n_upd, total, seed] + [sug[p.name] for p in spaces]
        t0 = time.time()
        try:
            import jax
            perf = float(run_experiment(cfg))
            jax.block_until_ready(perf)
            dt = time.time() - t0
            # Only observed trials are fed back; a failure is recorded but never
            # invented as a score, which would bias the surrogate.
            carbs.observe(ObservationInParam(input=sug, output=perf,
                                             cost=sug["num_updates"]))
            ok += 1
            if perf > best:
                best, best_sug = perf, dict(sug, _seed=seed, _num_updates=n_upd,
                                            _total=total, _wall=dt)
            row[1:4] = ["ok", f"{perf:.4f}", f"{dt:.1f}"]
            print(f"  [{t:2d}/{N_TRIALS}] return {perf:8.3f}  "
                  f"updates {n_upd:5d} ({total/1e6:5.2f}M)  {dt:6.1f}s   best {best:.3f}")
        except Exception:
            fail += 1
            row[1:4] = ["FAILED", "", f"{time.time()-t0:.1f}"]
            print(f"  [{t:2d}/{N_TRIALS}] FAILED\n{traceback.format_exc()}")
        tw.writerow(row)
        tf.flush()

    tf.close()
    print(f"  -> {ok} observed, {fail} failed. best return {best:.4f}")
    print(f"  -> trials: {trials_path}")
    return best, best_sug, ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algos", nargs="*", default=None,
                    help="subset of the three; default all")
    ap.add_argument("--seed", type=int, default=42, help="RNG for per-trial seeds")
    a = ap.parse_args()
    algos = a.algos or ALGOS
    bad = [x for x in algos if x not in ALGOS]
    if bad:
        sys.exit(f"unknown: {bad}\nchoose from: {ALGOS}")

    print("=== preflight ===")
    ubs, n_dev = preflight()
    rng = np.random.default_rng(a.seed)

    results = {}
    for algo in algos:
        results[algo] = sweep(algo, ubs, n_dev, rng)

    # Best row per algorithm, in the schema of matrix-games-carbs-sweep.csv.
    os.makedirs(OUT_DIR, exist_ok=True)
    best_path = os.path.join(OUT_DIR, "climbing-carbs-best.csv")
    with open(best_path, "w", newline="") as f:
        w = csv.DictWriter(f, SWEEP_COLS)
        w.writeheader()
        for algo, (best, sug, _ok, _fail) in results.items():
            if sug is None:
                continue
            w.writerow({
                "system_name": algo, "task": TASK, "env_name": ENV_NAME,
                "run_id": f"carbs-climbing-{algo}", "episode_return": round(best, 6),
                "win_rate": "", "time_series_total": sug["_total"],
                "wall_clock_time": round(sug["_wall"], 1),
                "num_minibatches": int(2 ** sug["num_minibatches"]),
                "vf_coef": sug["vf_coef"], "max_grad_norm": sug["max_grad_norm"],
                "decay_scaling_factor": "", "ent_coef": sug["ent_coef"],
                "actor_lr": sug["actor_lr"], "gae_lambda": sug["gae_lambda"],
                "critic_lr": sug["critic_lr"], "gamma": sug["gamma"],
                "ppo_epochs": sug["ppo_epochs"], "clip_eps": sug["clip_eps"],
                "num_envs": NUM_ENVS, "num_updates": sug["_num_updates"],
                "num_evaluation": int(sug["num_updates"]),
            })

    print(f"\n=== summary ===")
    for algo, (best, _s, ok, fail) in results.items():
        print(f"  {algo:42s} best {best:8.4f}   ({ok} ok, {fail} failed)")
    print(f"\n  best params -> {best_path}")
    print("  merge into optimal_parameters/matrix-games-carbs-sweep.csv, then add the")
    print("  three algorithms to run_climbing_rerun.sh to produce the tuned ladder.")


if __name__ == "__main__":
    main()
