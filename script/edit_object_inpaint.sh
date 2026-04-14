#!/usr/bin/env bash
#SBATCH --output=/share/sopio/slurm/%J.out
#SBATCH --error=/share/sopio/slurm/%J.err
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-0
#SBATCH --partition=3090
#SBATCH -J noconstr_direct


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
squeue

# Activate a conda environment if necessary.
source /data/sopio/miniconda3/bin/activate
conda activate gaussian_grouping

# # Check if the user provided an argument
# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <output_folder> <config_file> "
#     exit 1
# fi


# output_folder="$1"
# config_file="$2"
# if [ ! -d "$output_folder" ]; then
#     echo "Error: Folder '$output_folder' does not exist."
#     exit 2
# fi



# Remove the selected object
python edit_object_inpaint.py  -m output/small_city_50/25 --config_file config/object_inpaint/road_damage.json --inpaint_strategy "direct" --iteration 7000



