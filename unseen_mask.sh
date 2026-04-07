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

export CUDA_HOME=/data/sopio/miniconda3/envs/gaussian_grouping
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# cd /share/sopio/master_thesis/codebases/gaussian-grouping/Tracking-Anything-with-DEVA/Grounded-Segment-Anything/GroundingDINO
# pip install -e .

cd /share/sopio/master_thesis/codebases/gaussian-grouping/Tracking-Anything-with-DEVA/

img_path=/share/sopio/master_thesis/codebases/gaussian-grouping/output/small_city_50/25/train/ours_object_removal/iteration_30000/renders
mask_path=/share/sopio/master_thesis/codebases/gaussian-grouping/output/small_city_50/25/train/ours_object_removal/iteration_30000/gt_objects_color
lama_path=/share/sopio/master_thesis/codebases/gaussian-grouping/lama/LaMa_test_images/small_city_50/25

python demo/demo_with_text.py   --chunk_size 4    --img_path $img_path  --amp \
  --temporal_setting semionline --size 480   --output $mask_path  \
  --prompt "black blurry hole"

python prepare_lama_input.py $img_path $mask_path $lama_path
cd ..