# Neural Network Reconstruction of Neutrino Tracks in IceCube Observatory with Photon Propagation PDFs

An automated pipeline for reconstruction of muon neutrino tracks in IceCube Neutrino Observatory. Reconstruction is performed through tensor batches to enable batch processing. 

The processing scripts are currently in beta version. Further optimization and updating to the latest package are currently in the works. Read the software requirements very carefully and do not update to the latest version of the packages described below, as they could break the dependencies of the processing scripts. 

## Installation

Currently the processing and reconstruction scripts are based on JAX version=0.4.2.3 and TensorFlow version=2.1.5.0, and currently for the conversion scripts we are still using numpy=1.2.6.4. 

### Requirements

```bash
- Python==3.10.14
- conda create --name 3pandelnet python=3.10.14
- python -m pip uninstall -y jax jaxlib ml-dtypes numpy scipy

- python -m pip install "numpy==1.26.4" "scipy==1.12.0"
- python -m pip install --only-binary=:all: "ml-dtypes==0.2.0"

- python -m pip install "jax[cuda11_pip]==0.4.23" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

- python -m pip install tensorflow-cpu==2.15.0
- python -m pip install -U "equinox==0.11.10" "jaxtyping==0.3.3" --no-deps
- python -m pip install "optimistix==0.0.8" --no-deps
- python -m pip install "lineax==0.0.6" --no-deps
- python -m pip install "quadax==0.2.9" --no-deps
- python -m pip install --only-binary=:all: "pyarrow==14.0.2"
```

### Setup enviroment

```bash
git https://github.com/rishibbdb/I3-TrackPDFReco
cd I3-TrackPDFReco

# Source environment

conda activate 3pandelnet

- **For data pre-processing, use the following IceCube commands. Ignore if performing reconstruction. 

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

```
### Preprocessing Scripts

All pre-processing scripts are found in the `preprocessing` directory. 
- **convert_one_event.py: the script converts one event from an I3 file into Pandas data frame format for evaluation. 
- **convert_i3_ftr_coinc_muonlabel.py: this script converts all the events from all the i3 files in a directory and removes events that contain a coincident muon associated with that specific event. The files in the feather format can also be used for reconstruction sequentially and not in batch processing. 
- **convert_ftr_tfrecords.py: this script converts all events from the feather format to a TFRecords format compatible with TFRecordBatches for multiprocessing. 

`convert_i3_ftr_coinc_muonlabel.py` can take arguments as input, but `convert_ftr_tfrecords.py` currently is not implemented with arguments, which is currently in development. The usage of the conversion scripts from I3 to Feather and Feather to TFRecords with slurm jobs is as follows:

**Note: For preprocessing, you need to activate/setup the IceCube icetray environment. 

```bash

#!/bin/bash --login
#SBATCH --ntasks=1       # number of CPUs
#SBATCH --mem-per-cpu=80G # memory for CPUs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name 22644_job1

# module --force purge
### Setup icecube environment
module --force purge
module load CUDA/12.6.0
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`

ICETRAY_ENV="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/Ubuntu_22.04_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh"

### Setup TrackReco Environment
source /mnt/home/baburish/miniconda3/miniconda/etc/profile.d/conda.sh
export PYTHONPATH="/mnt/home/baburish/miniconda3/miniconda/envs/3pandelnet/lib/python3.10/site-packages/:$PYTHONPATH"
conda activate 3pandelnet

SCRIPT_DIR="/mnt/scratch/baburish/TPN-training/final/TPN_God/preprocessing"

DATASET_ID=22644
INDEX_START=0000
INDEX_END=1000
FILE_BASE=FinalLevel_NuMu_NuGenCCNC.022644
OUTPUT_DIR_FTR="/mnt/research/IceCube/Gupta-Reco/22644/0000000-0000999-tfrecords/"
OUTPUT_DIR_LOGS="/mnt/scratch/baburish/TPN-training/final/TPN_God/job_scripts/logs-22644/"
OUTPUT_DIR_TF="/mnt/research/IceCube/Gupta-Reco/22644/tfrecords/tf"
i3DIR='/mnt/research/IceCube/Gupta-Reco/22644/0000000-0000999/'


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


```
These processing scripts can be found in the `job_scripts` directory under their respective dataset name. 

### Reconstruction 

Reconstruction is performed using the `batch_reconstruction.py` file, and the respective slurm job script is in `batchreconstruction_job.sh`. 
**Note: Reconstruction does not use the IceCube environment. 

```bash
python3.10 /mnt/scratch/baburish/TPN-training/final/TPN_God/batch_reconstruction.py -f $path/to/tfrecords --seed $seed --outfile $path/to/results --batch_size $batch_size 
```

where
- **-f: path to the tfrecords files containing the events to be reconstructed. 
- **--seed: the initial seed of the track direction, by default and recommended, is `linefit`. Other options include `splinempe` which uses the splineMPE direction and the `truth` Which uses the true direction of the track. 

### Diagnostics

For plotting the performance of the network, there is a Jupyter notebook inside the `notebooks` directory call as the `angular_resolution.ipynb`
For plotting individual likelihood scans of the direction of the neutrino track, use the `test-linefit-seed.ipynb` script. 

## Performance Notes

- **Typical Runtime**: Typical runtime for reconstruction is two seconds per event for a batch reconstruction
- **Memory Usage**: Memory scales with the size of the batches. Recommended to use lower values in brackets 2 or 3 to optimize memory usage. 

## Support

For issues, questions, or suggestions:
- Open an [GitHub Issue](https://github.com/rishibbdb/I3-TrackPDFReco/issues)
- Contact: Rishi Babu (rbabu@mtu.edu, rbabu@icecube.wisc.edu)

## Version Tracking
- **alpha build: Preliminary tests with linefit
- **beta build(Current): Added batch reconstruction/Optimizations for likelihood analysis
- **gamma build(TBD): Add noise improvements, update to latest packages
