import glob
import sys, os
import argparse
import psutil

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file_path", type=str,
                  default="/home/storage2/hans/i3files/21217/ftr/",
                  dest="PATH_TO_INPUT",
                  help="directory containing the event data .tfrecords files")

parser.add_argument("-tf", "--tfrecords_file", type=str,
                  default="/home/storage2/hans/i3files/21217/ftr/data_ds_21217_from_*_to_*_1st_pulse.tfrecord",
                  dest="TFRECORDS_FILE_NAME",
                  help="Name of the .tfrecords files containing event meta data")

parser.add_argument("-g", "--gpu", type=int,
                  default=0,
                  dest="GPU_INDEX",
                  help="which GPU should run the code")

parser.add_argument("-b", "--batch_size", type=float,
                  default=1,
                  dest="BATCH_SIZE",
                  help="how many events should go into one batch")

parser.add_argument("-s", "--seed", type=str,
                    default="spline_mpe",
                    dest="SEED",
                    help="options are: spline_mpe, truth, linefit")

parser.add_argument("-nb", "--stop_after_n_batches", type=int,
                    default=100000000000,
                    dest="STOP_AFTER_N_BATCHES",
                    help="Set a small number if you want to test the script on a couple of batches")

parser.add_argument("-o", "--outfile", type=str,
                    default="results.npy",
                    dest="OUTFILE",
                    help="Where to write the reconstruction results")

# whether or not to shift the seed such that the vertex
# corresponds to the charge weighted median time of the event
parser.add_argument('--center_track_seed', default=False, action=argparse.BooleanOptionalAction)

# whether or not to use multiple vertex seeds: ~factor of 6 slower
parser.add_argument('--use_multiple_vertex_seeds', default=True, action=argparse.BooleanOptionalAction)

# whether or not to pre-scan the time axis to best-match the seed vertex.
parser.add_argument('--prescan_time', default=True, action=argparse.BooleanOptionalAction)

# Gaussian convolution widths for multi-stage fitting (in nanoseconds)
parser.add_argument('-gc', '--gaussian_conv_widths', type=float, nargs='+',
                    default=[100, 10, 3],
                    dest="GAUSSIAN_CONV_WIDTHS",
                    help="List of Gaussian convolution widths (ns) for iterative fitting stages")


args = parser.parse_args()
print(args)
print("")

import sys, os
sys.path.insert(0, "/mnt/scratch/baburish/TPN-training/final/TPN_God")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow_probability.substrates import jax as tfp
import tensorflow as tf

# Import JAX and require double precision.
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

# Other tools.
from scipy.interpolate import griddata
from lib.gupta import precompute_fixed_grid
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import csv
# Import TriplePandel stuff
from lib.simdata_i3 import I3SimBatchHandlerTFRecord, I3SimHandler
from lib.geo import center_track_pos_and_time_based_on_data, center_track_pos_and_time_based_on_data_batched_v
from lib.gupta_network_eqx_4comp import get_network_eval_v_fn, get_network_eval_v_fn_f32
from lib.experimental_methods import get_vertex_seeds
from lib.linefit import linefit_3d_time_np, linefit_3d_time_jnp
from fitting.llh_scanner import get_scanner
from fitting.llh_fitter import get_fitter
from lib.dom_track_eval import get_eval_network_doms_and_track
from lib.likelihood_conv_mpe_logsumexp_gupta import get_neg_c_triple_gamma_llh, get_neg_c_triple_gamma_llh_optimized
from lib.likelihood_conv_mpe_w_noise_logsumexp_gupta import get_neg_c_triple_gamma_llh_SRT_noise
from astropy.coordinates import SkyCoord

from palettable.cubehelix import Cubehelix
cx = Cubehelix.make(start=0.3, rotation=-0.5, n=16, reverse=False, gamma=1.0,
     	max_light=1.0,max_sat=0.5, min_sat=1.4).get_mpl_colormap()
import astropy.units as u
from helpers import *

def print_memory_usage(tag=""):
    # System RAM for this process
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    label = f" {tag}" if tag else ""
    print(f"[MEM{label}] Process RSS: {rss_gb:.2f} GB | "
          f"System: {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB ({vm.percent:.0f}%)")

    # GPU memory via JAX (None on some backends -> guarded)
    try:
        for dev in jax.devices():
            stats = dev.memory_stats()
            if stats:
                print(f"         GPU {dev.id}: in_use "
                      f"{stats.get('bytes_in_use', 0)/1e9:.2f} GB | "
                      f"peak {stats.get('peak_bytes_in_use', 0)/1e9:.2f} GB | "
                      f"limit {stats.get('bytes_limit', 0)/1e9:.2f} GB")
    except Exception as e:
        print(f"         (GPU memory stats unavailable: {e})")


dzen = 0.05 # rad
dazi = 0.05 # rad
n_eval = 25 # number of grid points per axes



n_hidden = 96
gupta = True
n_comp = 4

network_path = '/mnt/scratch/baburish/TPN-training/gupta_mixture_jax/new_weights/4comp_no_penalties_w4096batch_tree_start_epoch_255.eqx'
eval_network_v = get_network_eval_v_fn_f32(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=gupta, n_comp=n_comp)
print_memory_usage("after network load")

if '*' in args.TFRECORDS_FILE_NAME:
    fs = glob.glob(os.path.join(args.PATH_TO_INPUT, args.TFRECORDS_FILE_NAME))
else:
    fs = [os.path.join(args.PATH_TO_INPUT, args.TFRECORDS_FILE_NAME)]  # ← Add this line!

batch_maker = I3SimBatchHandlerTFRecord(fs, batch_size=args.BATCH_SIZE, n_labels=20)
# Create padded batches (with different seq length).
batch_iter = batch_maker.get_batch_iterator()
# Check total number of events loaded
print(f"\n{'='*60}")
print("Checking total events in dataset...")
print(f"{'='*60}")

total_events_available = 0
for pulse_data, meta_data in batch_iter:
    total_events_available += pulse_data.shape[0]

print(f"Total events available in dataset: {total_events_available}")
print(f"{'='*60}\n")

batch_iter = batch_maker.get_batch_iterator()
# Multi-stage fitting function

staged_fitters = []
for sigma in args.GAUSSIAN_CONV_WIDTHS:
    neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sigma=sigma)
    fit_llh = get_fitter(
        neg_llh,
        use_multiple_vertex_seeds=args.use_multiple_vertex_seeds,
        prescan_time=args.prescan_time,
        use_batches=True,
    )
    staged_fitters.append((sigma, jax.jit(fit_llh)))


def fit_batch_with_sigma_stages(track_src, centered_track_pos, centered_track_time, pulse_data):
    best_logl = best_direction = best_vertex = best_time = None
    for sigma, fit_llh_jit in staged_fitters:
        print(f"  Fitting with Gaussian convolution width: {sigma} ns")
        if best_logl is not None:
            seed_src, seed_pos, seed_time = best_direction, best_vertex, best_time
        else:
            seed_src, seed_pos, seed_time = track_src, centered_track_pos, centered_track_time
        best_logl, best_direction, best_vertex, best_time = fit_llh_jit(
            seed_src, seed_pos, seed_time, pulse_data
        )
    return best_logl, best_direction, best_vertex, best_time
# def fit_batch_with_sigma_stages(track_src, centered_track_pos, centered_track_time, pulse_data):
#     """
#     Perform iterative fitting across multiple Gaussian convolution widths.
#     Uses output from one stage as seed for the next.
    
#     Returns: (logl, direction, vertex, time)
#     """
#     best_logl = None
#     best_direction = None
#     best_vertex = None
#     best_time = None
    
#     for sigma in args.GAUSSIAN_CONV_WIDTHS:
#         print(f"  Fitting with Gaussian convolution width: {sigma} ns")
        
#         # Setup likelihood with specified gaussian convolution width
#         neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sigma=sigma)
        
#         # Setup fitter with improved configuration
#         fit_llh = get_fitter(
#             neg_llh,
#             use_multiple_vertex_seeds=args.use_multiple_vertex_seeds,
#             prescan_time=args.prescan_time,
#             use_batches=True
#         )
        
#         # JIT compile for this batch
#         fit_llh_jit = jax.jit(fit_llh)
        
#         # Use previous stage's result as seed, or initial seed for first stage
#         if best_logl is not None:
#             seed_src = best_direction
#             seed_pos = best_vertex
#             seed_time = best_time
#         else:
#             seed_src = track_src
#             seed_pos = centered_track_pos
#             seed_time = centered_track_time
        
#         # Run the fit
#         solution = fit_llh_jit(seed_src, seed_pos, seed_time, pulse_data)
#         current_logl, current_direction, current_vertex, current_time = solution

#         best_logl = current_logl
#         best_direction = current_direction
#         best_vertex = current_vertex
#         best_time = current_time
    
#     return best_logl, best_direction, best_vertex, best_time


# Process batches

column_names = [
    'neutrino_energy', 'q_tot', 'muon_zenith', 'muon_azimuth', 'muon_time',
    'muon_pos_x', 'muon_pos_y', 'muon_pos_z',
    'spline_mpe_zenith', 'spline_mpe_azimuth', 'spline_mpe_time',
    'spline_mpe_pos_x', 'spline_mpe_pos_y', 'spline_mpe_pos_z',
    'linefit_zenith', 'linefit_azimuth', 'linefit_time',
    'linefit_pos_x', 'linefit_pos_y', 'linefit_pos_z',
    'reco_logl', 'reco_zenith', 'reco_azimuth',
    'reco_pos_x', 'reco_pos_y', 'reco_pos_z', 'reco_time',
]

out_csv = f"{args.OUTFILE}.csv"
# Fresh file each run; remove any stale output so we don't append to old data.
if os.path.exists(out_csv):
    os.remove(out_csv)

header_written = False
total_events_written = 0

collect_results = []
finished_batches = False
batch_idx = 0
total_start_time = time.time()



while not finished_batches:
    if batch_idx == args.STOP_AFTER_N_BATCHES:
        print(f"Stopping early per user request (--stop_after_n_batches {args.STOP_AFTER_N_BATCHES}). Hence, did not reconstruct all available batches.")
        break

    try:
        print(f"\n{'='*60}")
        print(f"Reconstructing batch {batch_idx}")
        print(f"{'='*60}")
        
        batch_start_time = time.time()
        pulse_data, meta_data = batch_iter.next() # [Nev, Ndom, Nobs], [Nev, Naux]
        pulse_data = jnp.array(pulse_data.numpy())
        meta_data = jnp.array(meta_data.numpy())

        if args.SEED == "spline_mpe":
            seed_data = meta_data[:, 8:14]
        elif args.SEED == "truth":
            seed_data = meta_data[:, 2:8]
        
        elif args.SEED == "linefit":
            seed_data = meta_data[:, 14:20]
        else:
            raise ValueError(f"seed {args.SEED} not available. Use spline_mpe, truth, or linefit")

        seed_data = jnp.array(seed_data)
        centered_track_pos, centered_track_time = seed_data[:, 3:], seed_data[:, 2]
        track_src = seed_data[:, :2]

        if args.center_track_seed:
            print("Shifting seed vertex based on charge-weighted median time...")
            centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data_batched_v(pulse_data, seed_data)

        print(f"Seed vertex of first event in batch: {centered_track_pos[0]} m")
        print(f"Data shape: {pulse_data.shape}")
        print(f"Number of events in batch: {pulse_data.shape[0]}")

        # Fit batch with multi-stage Gaussian convolution widths
        logl, direction, vertex, track_time = fit_batch_with_sigma_stages(
            track_src, centered_track_pos, centered_track_time, pulse_data
        )
        out_data = jnp.concatenate(
            [
                meta_data,
                jnp.expand_dims(logl, axis=1),
                direction,
                vertex,
                jnp.expand_dims(track_time, axis=1)
            ],
            axis=1
        )

        # collect_results.append(out_data)
        df_batch = pd.DataFrame(np.array(out_data), columns=column_names)
        df_batch.to_csv(out_csv, mode='a', header=not header_written, index=False)
        header_written = True
        total_events_written += out_data.shape[0]
        print(f"  Wrote {out_data.shape[0]} events (total so far: {total_events_written})")
        
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx} completed in {batch_time:.2f}s")
        print_memory_usage("after network load")
        batch_idx += 1

    except StopIteration:
        print("\nFinished processing all available batches.")
        finished_batches = True

# Concatenate results from all batches
print(f"\n{'='*60}")
print("Finalizing results...")
print(f"{'='*60}")
print(f"\nTotal events processed and written: {total_events_written}")
print(f"✓ Saved: {out_csv}")
# results = jnp.concatenate(collect_results, axis=0)
# print_memory_usage("after network load")


# print(f"\nTotal events processed: {results.shape[0]}")
# print(f"Result array shape: {results.shape}")

# # Convert to pandas DataFrame
# column_names = [
#     'neutrino_energy', 'q_tot', 'muon_zenith', 'muon_azimuth', 'muon_time',
#     'muon_pos_x', 'muon_pos_y', 'muon_pos_z',
#     'spline_mpe_zenith', 'spline_mpe_azimuth', 'spline_mpe_time',
#     'spline_mpe_pos_x', 'spline_mpe_pos_y', 'spline_mpe_pos_z',
#     'linefit_zenith', 'linefit_azimuth', 'linefit_time',  # ← Add these (14-16)
#     'linefit_pos_x', 'linefit_pos_y', 'linefit_pos_z',    # ← Add these (17-19)
#     'reco_logl', 'reco_zenith', 'reco_azimuth',
#     'reco_pos_x', 'reco_pos_y', 'reco_pos_z', 'reco_time',
# ]

# results_np = np.array(results)
# df = pd.DataFrame(results_np, columns=column_names)

# print(f"\nSaving results to {args.OUTFILE}")

# # CSV (human-readable)
# df.to_csv(f"{args.OUTFILE}.csv", index=False)
print(f"✓ Saved: {args.OUTFILE}.csv")

total_elapsed = time.time() - total_start_time
print(f"\nTotal processing time: {total_elapsed:.2f}s")
print("Done!")