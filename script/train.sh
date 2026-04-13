#!/usr/bin/env bash
#SBATCH --output=/share/sopio/slurm/%J.out
#SBATCH --error=/share/sopio/slurm/%J.err
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-0
#SBATCH --partition=3090
#SBATCH -J nonconstr


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
squeue

# Activate a conda environment if necessary.
source /data/sopio/miniconda3/bin/activate
conda activate gaussian_grouping

# Check if the user provided an argument
# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <dataset_name>"
#     exit 1
# fi


# dataset_name="$1"
# scale="$2"
# dataset_folder="$dataset_name"

# if [ ! -d "$dataset_folder" ]; then
#     echo "Error: Folder '$dataset_folder' does not exist."
#     exit 2
# fi
# python process.py

# pip install viser



# python inspect_ply.py /data/sopio/small_city_50/25/sparse_25/0/points3D_original.ply
# /share/sopio/master_thesis/codebases/gaussian-grouping/output/small_city_50/25/input.ply

# python visualize_pc_axes.py /share/sopio/master_thesis/codebases/gaussian-grouping/output/small_city_50/25/input.ply

# Gaussian Grouping training
python train.py    -s /data/sopio/small_city_50/25 -r 1  -m output/small_city_50/25_2_objects --config_file config/gaussian_dataset/train.json --iteration 7000

# Segmentation rendering using trained model
python render.py -m output/small_city_50/25_2_objects --num_classes 3
