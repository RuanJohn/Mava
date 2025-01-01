#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="rec-benchmark-humanoid"
#SBATCH --mail-user=dkcrua001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12

source /home/dkcrua001/Mava/.venv/bin/activate

cd Mava

python mava/systems/ppo/anakin/rec_ppo_central.py -m system.seed=0,1,2,3,4,5,6,7,8,9 \
env=mabrax \
system.actor_lr=0.0005  \
system.critic_lr=0.0005  \
system.clip_eps=0.1 \
system.ent_coef=0.0 \
system.max_grad_norm=5.0 \
system.num_minibatches=8 \
system.ppo_epochs=4 \
system.recurrent_chunk_size=64 \
env.scenario.name="humanoid_9|8" env.scenario.task_name="humanoid_9|8"
