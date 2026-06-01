#!/bin/bash --login
#SBATCH --ntasks=1       # number of CPUs
#SBATCH --mem-per-cpu=80G # memory for CPUs
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name 22645_job1

module --force purge
module load CUDA/12.6.0
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`

# /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh
ICETRAY_ENV="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh"

source /mnt/home/baburish/miniconda3/miniconda/etc/profile.d/conda.sh
export PYTHONPATH="/mnt/home/baburish/miniconda3/miniconda/envs/3pandelnet/lib/python3.10/site-packages/:$PYTHONPATH"
conda activate 3pandelnet

SCRIPT_DIR="/mnt/scratch/baburish/TPN-training/final/TPN_God/preprocessing"

DATASET_ID=22645
INDEX_START=0000
INDEX_END=1000
FILE_BASE=FinalLevel_NuMu_NuGenCCNC.022645
OUTPUT_DIR_FTR="/mnt/research/IceCube/Gupta-Reco/22645/0000000-0000999-tfrecords/"
OUTPUT_DIR_LOGS="/mnt/scratch/baburish/TPN-training/final/TPN_God/job_scripts/logs-22645/"
OUTPUT_DIR_TF="/mnt/research/IceCube/Gupta-Reco/22645/tfrecords/tf"
i3DIR='/mnt/research/IceCube/Gupta-Reco/22645/0000000-0000999/'


if [ ! -d "$OUTPUT_DIR_FTR" ]; then
  mkdir -p "$OUTPUT_DIR_FTR"
fi
if [ ! -d "$OUTPUT_DIR_LOGS" ]; then
  mkdir -p "$OUTPUT_DIR_LOGS"
fi
if [ ! -d "$OUTPUT_DIR_TF" ]; then
  mkdir -p "$OUTPUT_DIR_TF"
fi

# Run the conversion script
"$ICETRAY_ENV" "$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/convert_i3_ftr_coinc_muonlabel.py" --i3dir $i3DIR --dataset-id $DATASET_ID --file-index-start $INDEX_START --file-index-end $INDEX_END --outdir-ftr $OUTPUT_DIR_FTR --file-base $FILE_BASE | tee $OUTPUT_DIR_LOGS/job1log.txt

