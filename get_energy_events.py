

import sys, os
sys.path.insert(0, "/mnt/scratch/baburish/TPN-training/final/TPN_God")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorflow_probability.substrates import jax as tfp
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
from dom_track_eval import get_eval_network_doms_and_track
from lib.likelihood_conv_mpe_logsumexp_gupta import get_neg_c_triple_gamma_llh, get_neg_c_triple_gamma_llh_optimized
from lib.likelihood_conv_mpe_w_noise_logsumexp_gupta import get_neg_c_triple_gamma_llh_SRT_noise
from astropy.coordinates import SkyCoord

from palettable.cubehelix import Cubehelix
cx = Cubehelix.make(start=0.3, rotation=-0.5, n=16, reverse=False, gamma=1.0,
     	max_light=1.0,max_sat=0.5, min_sat=1.4).get_mpl_colormap()
import astropy.units as u
from helpers import *

dzen = 0.05 # rad
dazi = 0.05 # rad
n_eval = 25 # number of grid points per axes



n_hidden = 96
gupta = True
n_comp = 4

network_path = '/mnt/scratch/baburish/TPN-training/gupta_mixture_jax/new_weights/4comp_no_penalties_w4096batch_tree_start_epoch_255.eqx'
eval_network_v = get_network_eval_v_fn_f32(bpath=network_path, dtype=dtype, n_hidden=n_hidden)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype, gupta=gupta, n_comp=n_comp)
PATH_TO_INPUT = '/mnt/research/IceCube/Gupta-Reco/22645/tfrecords/ftr/'
META_FILE_NAME = 'meta_ds_22645_from_0_to_1000_10_to_100TeV.ftr'
PULSES_FILE_NAME = 'pulses_ds_22645_from_0_to_1000_10_to_100TeV.ftr'


events_meta_file = os.path.join(PATH_TO_INPUT, META_FILE_NAME)
events_pulses_file = os.path.join(PATH_TO_INPUT, PULSES_FILE_NAME)
geo_file = '/mnt/scratch/baburish/TPN-training/TriplePandelReco_JAX/data/icecube/detector_geometry.csv'

events_meta = pd.read_feather(events_meta_file)
events_data = pd.read_feather(events_pulses_file)
geo = pd.read_csv(geo_file)


int_cols_meta = ["event_id", "idx_start", "idx_end", "n_channel_HLC", "n_channel"]
events_meta[int_cols_meta] = events_meta[int_cols_meta].astype("Int64")

int_cols_data = ["event_id", "sensor_id", "is_HLC"]
events_data[int_cols_data] = events_data[int_cols_data].astype("Int64")
print("Running reconstruction for events in file:", events_meta_file)
with open("/mnt/scratch/baburish/TPN-training/final/TPN_God/reco-logs/22645_255epoch_linefit_iterative_energy.csv", "a", newline="") as f:
    writer = csv.writer(f)
    print("Writing results to reco-logs/22645_255epoch_linefit_iterative_energy.csv")
    writer.writerow(["event_id", "neutrino_energy", "muon_energy"])
    for i in range(0,len(events_meta)):
        start = time.time()
        print(f"Running reconstruction for event id: {i}")
        EVENT_INDEX=i
        meta, pulses = get_event_data(EVENT_INDEX, events_meta, events_data)
        print(f"Neutrino energy: {meta['muon_energy']/1.e3:.1f} TeV")
        print("Muon Energy in Detector:", meta['muon_energy_at_detector']/1.e3, "TeV")

        writer.writerow([
            i,
            meta["muon_energy"]/1.e3,
            meta['muon_energy_at_detector']/1.e3,
        ])
        f.flush()
        elapsed = time.time() - start
        print(f"Time elapsed: {elapsed:.2f}s")
        print("--------------------------------------------------")
