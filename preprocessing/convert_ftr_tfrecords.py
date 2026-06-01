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
from lib.simdata_i3 import I3SimHandler
from lib.geo import center_track_pos_and_time_based_on_data
from lib.gupta_network_eqx_4comp import get_network_eval_v_fn, get_network_eval_v_fn_f32
from lib.experimental_methods import get_vertex_seeds
from lib.linefit import linefit_3d_time_np, linefit_3d_time_jnp
from fitting.llh_scanner import get_scanner
from fitting.llh_fitter import get_fitter
from lib.dom_track_eval import get_eval_network_doms_and_track
from lib.likelihood_conv_mpe_logsumexp_gupta import get_neg_c_triple_gamma_llh, get_neg_c_triple_gamma_llh_optimized
from lib.likelihood_conv_mpe_w_noise_logsumexp_gupta import get_neg_c_triple_gamma_llh_SRT_noise
from pulse_extraction_from_i3 import get_pulse_info
from tfrecords_utils import serialize_example


from astropy.coordinates import SkyCoord

from palettable.cubehelix import Cubehelix
cx = Cubehelix.make(start=0.3, rotation=-0.5, n=16, reverse=False, gamma=1.0,
     	max_light=1.0,max_sat=0.5, min_sat=1.4).get_mpl_colormap()
import astropy.units as u
from helpers import *


import glob

PATH_TO_INPUT = '/mnt/research/IceCube/Gupta-Reco/22646/tfrecords'
META_FILE_NAME = 'meta_ds_22646_from_0_to_1000_10_to_100TeV.ftr'
PULSES_FILE_NAME = 'pulses_ds_22646_from_0_to_1000_10_to_100TeV.ftr'

events_meta_file = os.path.join(PATH_TO_INPUT, META_FILE_NAME)
events_pulses_file = os.path.join(PATH_TO_INPUT, PULSES_FILE_NAME)
geo_file = '/mnt/scratch/baburish/TPN-training/TriplePandelReco_JAX/data/icecube/detector_geometry.csv'

events_meta = pd.read_feather(events_meta_file)
events_data = pd.read_feather(events_pulses_file)

geo = pd.read_csv(geo_file)

# for i in range(len(events_meta)):
#     int_cols_meta = ["event_id", "idx_start", "idx_end", "n_channel_HLC", "n_channel"]
#     events_meta[int_cols_meta] = events_meta[int_cols_meta].astype("Int64")

#     int_cols_data = ["event_id", "sensor_id", "is_HLC"]
#     events_data[int_cols_data] = events_data[int_cols_data].astype("Int64")

#     meta, pulses = get_event_data(i, events_meta, events_data)

#     event_data = get_per_dom_summary_from_sim_data(meta, pulses, geo)
#     replace_early_pulse(event_data, pulses)
    # Get MCTruth.
    # true_pos = jnp.array([meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']])
    # true_time = meta['muon_time']
    # true_zenith = meta['muon_zenith']
    # true_azimuth = meta['muon_azimuth']
    # true_src = jnp.array([true_zenith, true_azimuth])
    # true_src_deg = np.rad2deg(true_src)
    # splinempe_zenith = meta['spline_mpe_zenith']
    # splinempe_azimuth = meta['spline_mpe_azimuth']
    # spline_src = jnp.array([splinempe_zenith, splinempe_azimuth])

    # track_pos, track_time, _, track_src = linefit(event_data)
    # track_pos  = jnp.asarray(track_pos)      # (3,)
    # track_time = jnp.asarray(track_time)     # scalar ()
    # track_src  = jnp.asarray(track_src)      # (2,)
    # track_zenith = float(track_src[0])
    # track_azimuth = float(track_src[1])
    # track_zenith_deg = np.degrees(track_zenith)
    # track_azimuth_deg = np.degrees(track_azimuth)
    # seed_ang_err = angular_separation_deg(
    #             true_src_deg[0], true_src_deg[1],
    #             track_zenith_deg, track_azimuth_deg
    #         )
    # print(f"  Seed vs True Angular Distance error from the ftr files: {seed_ang_err:.2f} deg")








outdir = "/mnt/research/IceCube/Gupta-Reco/22646/tfrecords/"
dataset_id = 22646
file_index_start = 0
file_index_end = 1000
compression_type = ''
options = tf.io.TFRecordOptions(compression_type=compression_type)

sim_handler = I3SimHandler(df_meta=events_meta,
                           df_pulses=events_data,
                           geo_file=geo_file)

write_path = os.path.join(outdir, f"data_ds_{dataset_id}_from_{file_index_start}_to_{file_index_end}_1st_pulse.tfrecord")

print(f"Converting {len(events_meta)} events to tfrecord with linefit seeds...")
print(f"Output: {write_path}\n")

n_events = len(events_meta)
n_failed = 0
start_time = time.time()

with tf.io.TFRecordWriter(write_path, options) as writer:
    for i in range(n_events):
        try:
            # meta, pulses = sim_handler.get_event_data(i)
            meta, pulses = get_event_data(i, events_meta, events_data)
            
            # Get dom locations, first hit times, and total charges
            # event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)
            event_data = get_per_dom_summary_from_sim_data(meta, pulses, geo)
            replace_early_pulse(event_data, pulses)
            # Run linefit
            track_pos, track_time, _, track_src = linefit(event_data)
            track_pos = np.asarray(track_pos)     # (3,)
            track_time = float(track_time)         # scalar
            track_src = np.asarray(track_src)     # (2,)
            
            linefit_zenith = float(track_src[0])
            linefit_azimuth = float(track_src[1])
            linefit_pos_x = float(track_pos[0])
            linefit_pos_y = float(track_pos[1])
            linefit_pos_z = float(track_pos[2])
            linefit_zenith_deg = np.degrees(linefit_zenith)
            linefit_azimuth_deg = np.degrees(linefit_azimuth)

            # true_zenith = meta['muon_zenith']
            # true_azimuth = meta['muon_azimuth']
            # true_src = jnp.array([true_zenith, true_azimuth])
            # true_src_deg = np.rad2deg(true_src)

            # seed_ang_err = angular_separation_deg(
            #     true_src_deg[0], true_src_deg[1],
            #     linefit_zenith_deg, linefit_azimuth_deg
            # )
            # print(f"  Seed vs True Angular Distance error: {seed_ang_err:.2f} deg")
            # Pulse data
            x = event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy()
            
            # Original metadata (15 values)
            y_meta = meta[['neutrino_energy', 'q_tot', 'muon_zenith', 'muon_azimuth', 'muon_time',
                          'muon_pos_x', 'muon_pos_y', 'muon_pos_z', 'spline_mpe_zenith',
                          'spline_mpe_azimuth', 'spline_mpe_time', 'spline_mpe_pos_x',
                          'spline_mpe_pos_y', 'spline_mpe_pos_z']].to_numpy()
            
            # Linefit results (6 values)
            y_linefit = np.array([linefit_zenith, linefit_azimuth, track_time,
                                 linefit_pos_x, linefit_pos_y, linefit_pos_z])
            
            # Concatenate: original (15) + linefit (6) = 21 total
            y = np.concatenate([y_meta, y_linefit])
            
            # Write to tfrecord
            writer.write(serialize_example(
                tf.constant(x, dtype=tf.float64),
                tf.constant(y, dtype=tf.float64),
            ))
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_events - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:4d}/{n_events}] ({rate:.1f} events/s, ~{remaining:.1f}s remaining)")
        
        except Exception as e:
            n_failed += 1
            print(f"  WARNING: Event {i} failed - {e}")
            continue

elapsed = time.time() - start_time
print(f"\n✓ Stored {n_events - n_failed}/{n_events} events in {write_path}")
print(f"  Failed events: {n_failed}")
print(f"  Time: {elapsed:.1f}s ({elapsed/n_events:.3f}s per event)")
print(f"\nMetadata format (21 values):")
print(f"  [0-14]:  Original metadata (15 values)")
print(f"  [15]:    linefit_zenith")
print(f"  [16]:    linefit_azimuth")
print(f"  [17]:    linefit_time")
print(f"  [18]:    linefit_pos_x")
print(f"  [19]:    linefit_pos_y")
print(f"  [20]:    linefit_pos_z")