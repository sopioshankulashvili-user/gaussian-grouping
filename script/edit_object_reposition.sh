#!/usr/bin/env bash
#SBATCH --output=/share/sopio/slurm/%J.out
#SBATCH --error=/share/sopio/slurm/%J.err
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-0
#SBATCH --partition=2080
#SBATCH -J reposition


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
squeue

# Activate a conda environment if necessary.
source /data/sopio/miniconda3/bin/activate
conda activate gaussian_grouping


# Remove the selected object
python edit_object_reposition.py  -m output/small_city_50/25 --config_file config/object_reposition/road_damage.json --iteration 15000
