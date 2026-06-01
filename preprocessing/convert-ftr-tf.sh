#!/bin/bash --login
#SBATCH --ntasks=1       # number of CPUs
#SBATCH --mem-per-cpu=60G # memory for CPUs
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name 22644_job1

# module --force purge
### Setup icecube environment
module --force purge
module load CUDA/12.6.0
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`

# /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh
ICETRAY_ENV="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh"

source /mnt/home/baburish/miniconda3/miniconda/etc/profile.d/conda.sh
export PYTHONPATH="/mnt/home/baburish/miniconda3/miniconda/envs/3pandelnet/lib/python3.10/site-packages/:$PYTHONPATH"
conda activate 3pandelnet

SCRIPT_DIR="/mnt/scratch/baburish/TPN-training/final/TPN_God/preprocessing"

OUTPUT_DIR_LOGS="/mnt/scratch/baburish/TPN-training/final/TPN_God/logs/"

# Run the conversion script
"$ICETRAY_ENV" "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/convert_ftr_tfrecords.py"| tee $OUTPUT_DIR_LOGS/22456-0to1000-ftr-to-tfrecords.txt

