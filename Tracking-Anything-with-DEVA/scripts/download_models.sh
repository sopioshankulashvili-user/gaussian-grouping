#!/usr/bin/env bash
#SBATCH --output=/share/sopio/slurm/%J.out
#SBATCH --error=/share/sopio/slurm/%J.err
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-0
#SBATCH --partition=2080
#SBATCH -J grouping


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
squeue

# Activate a conda environment if necessary.
source /data/sopio/miniconda3/bin/activate
conda activate gaussian_grouping


wget -P ./saves/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
wget -P ./saves/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P ./saves/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P ./saves/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt
wget -P ./saves/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py