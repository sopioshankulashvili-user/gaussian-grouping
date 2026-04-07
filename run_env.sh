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
# conda activate base

# echo "Using conda env: $CONDA_PREFIX"
# echo "Current dir: ${PWD}"


# conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping

rm -rf submodules/*/build/ submodules/*/dist/ submodules/*/*.egg-info

export CUDA_HOME=/data/sopio/miniconda3/envs/gaussian_grouping
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH


# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cuda-toolkit=12.4 -c pytorch -c nvidia
# pip install plyfile==0.8.1
# pip install tqdm scipy wandb opencv-python scikit-learn lpips

conda install -c conda-forge gcc=9.4.0 gxx=9.4.0 ninja libxcrypt

export TORCH_CUDA_ARCH_LIST="7.5 8.6"
export LDFLAGS="-L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu"
pip uninstall diff-gaussian-rasterization simple-knn -y
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn


# python convert.py -s /data/sopio/datasets/annotated_small_city_45/45