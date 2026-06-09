#!/bin/bash --login
#SBATCH --gpus=a100:1   # number of GPUs
#SBATCH --mem-per-gpu=40G # memory for CPUs
#SBATCH --time=12:01:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name 22646-TPN-God
#SBATCH --output=/mnt/scratch/baburish/TPN-training/final/TPN_God/reco-logs/batchreco-22646slurm_run0to1000.log
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

# python3.10 /mnt/scratch/baburish/TPN-training/final/TPN_God/batchreco.py -f /mnt/research/IceCube/Gupta-Reco/22645/tfrecords/ -tf data_ds_22645_from_0_to_1000_1st_pulse.tfrecord --seed linefit --outfile /mnt/scratch/baburish/TPN-training/final/TPN_God/results/result_22645_0_to_1000 --batch_size 2 | tee /mnt/scratch/baburish/TPN-training/final/TPN_God/results/22645_0_to_1000.log

python3.10 /mnt/scratch/baburish/TPN-training/final/TPN_God/batch_reconstruction.py -f /mnt/research/IceCube/Gupta-Reco/22646/tfrecords/ -tf data_ds_22646_from_0_to_1000_1st_pulse.tfrecord --seed linefit --outfile /mnt/scratch/baburish/TPN-training/final/TPN_God/results/result_22646_0_to_1000 --batch_size 2 | tee /mnt/scratch/baburish/TPN-training/final/TPN_God/results/22646_0_to_1000.log