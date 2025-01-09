#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --job-name="sweep-medium-4ag"
#SBATCH --mail-user=dkcrua001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12

source /home/dkcrua001/Mava/.venv/bin/activate

cd Mava

python mava/systems/ppo/anakin/ff_ppo_central_factored.py -m env=rware\
env/scenario=medium-4ag
