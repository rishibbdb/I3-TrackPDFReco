#!/bin/bash --login
#SBATCH --ntasks=8       # number of CPUs
#SBATCH --gpus=a100:1   # number of GPUs
#SBATCH --mem-per-gpu=20G # memory for CPUs
#SBATCH --time=12:59:59
#SBATCH --job-name 22646-TPN-God
#SBATCH --output=/mnt/scratch/baburish/TPN-training/final/TPN_God/logs/22646slurm.log
#SBATCH --mail-user=rbabu@mtu.edu
#SBATCH --mail-type=ALL

module load CUDA/12.6.0
source /mnt/home/baburish/miniconda3/miniconda/etc/profile.d/conda.sh
export PYTHONPATH="/mnt/home/baburish/miniconda3/miniconda/envs/3pandelnet/lib/python3.10/site-packages/:$PYTHONPATH"
conda activate 3pandelnet

export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import jax; print(jax.devices())"
/mnt/home/baburish/miniconda3/miniconda/envs/3pandelnet/bin/python /mnt/scratch/baburish/TPN-training/final/TPN_God/reconstruct22646.py | tee /mnt/scratch/baburish/TPN-training/final/TPN_God/logs/22646reco.log