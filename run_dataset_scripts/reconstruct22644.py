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
PATH_TO_INPUT = '/mnt/research/IceCube/Gupta-Reco/22644/tfrecords/ftr/'
META_FILE_NAME = 'meta_ds_22644_from_1000_to_2000_10_to_100TeV.ftr'
PULSES_FILE_NAME = 'pulses_ds_22644_from_1000_to_2000_10_to_100TeV.ftr'


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
with open("/mnt/scratch/baburish/TPN-training/final/TPN_God/reco-logs/22644_255epoch_linefit_iterative.csv", "a", newline="") as f:
    writer = csv.writer(f)
    print("Writing results to reco-logs/iterative-event_summary-terminal-smallbatch.csv")
    writer.writerow(["event_id", "neutrino_energy", "true dir", "linefit best dir", "linefit seed dir", "linefit_ang_err", "splinempe best dir", "splinempe_ang_err"])
    for i in range(0,len(events_meta)):
        start = time.time()
        print(f"Running reconstruction for event id: {i}")
        EVENT_INDEX=i
        meta, pulses = get_event_data(EVENT_INDEX, events_meta, events_data)
        print(f"Neutrino energy: {meta['neutrino_energy']/1.e3:.1f} TeV")
        print("Muon Energy in Detector:", meta['muon_energy_at_detector']/1.e3, "TeV")
        
        event_data = get_per_dom_summary_from_sim_data(meta, pulses, geo)
        replace_early_pulse(event_data, pulses)

        # Get MCTruth.
        true_pos = jnp.array([meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']])
        true_time = meta['muon_time']
        true_zenith = meta['muon_zenith']
        true_azimuth = meta['muon_azimuth']
        true_src = jnp.array([true_zenith, true_azimuth])
        true_src_deg = np.rad2deg(true_src)
        splinempe_zenith = meta['spline_mpe_zenith']
        splinempe_azimuth = meta['spline_mpe_azimuth']
        spline_src = jnp.array([splinempe_zenith, splinempe_azimuth])
        print('Total charge in event:', event_data['charge'].sum())
        
        track_pos, track_time, _, track_src = linefit(event_data)
        track_pos  = jnp.asarray(track_pos)      # (3,)
        track_time = jnp.asarray(track_time)     # scalar ()
        track_src  = jnp.asarray(track_src)      # (2,)

        center_track_seed = True

        centered_track_pos, centered_track_time = track_pos, track_time
        if center_track_seed:
            print("shifting seed vertex.")
            centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data(event_data, track_pos, track_time, track_src)
        fitting_event_data = jnp.array(event_data[['x', 'y', 'z', 'time', 'charge']].to_numpy())
        GAUS_CONV_WIDTH = [100, 10, 3]
        best_logl = None
        best_logl = None
        # for sigma in GAUS_CONV_WIDTH:
        #     print(f"\nStarting fit with Gaussian convolution width: {sigma} ns")
        #     neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sigma=sigma)
        #     fit_llh = get_fitter(neg_llh, use_multiple_vertex_seeds=False, prescan_time=True)
        #     fit_llh_jit = jax.jit(fit_llh)
            
        #     if best_logl is not None:
        #         seed_src = best_direction
        #         seed_pos = best_vertex
        #         seed_time = best_time
        #     else:
        #         seed_src = track_src
        #         seed_pos = centered_track_pos
        #         seed_time = centered_track_time
            
        #     solution = fit_llh_jit(seed_src, seed_pos, seed_time, fitting_event_data)
        #     current_logl, current_direction, current_vertex, current_time = solution

        #     if best_logl is None:
        #         best_logl = current_logl
        #         best_direction = current_direction
        #         best_vertex = current_vertex
        #         best_time = current_time
        #         accepted = True
        #         print(f"Initial fit: logl {current_logl:.3f}")
        #         continue  
        #     else:
        #         print(f"Current fit: logl {best_logl:.3f}")
        #     delta_logl = -2*(current_logl - best_logl)
        #     if meta['q_tot'] > 100:
        #         best_logl = current_logl
        #         best_direction = current_direction
        #         best_vertex = current_vertex
        #         best_time = current_time
        #     else:
        #         if sigma <= 2:
        #             best_logl = current_logl
        #             best_direction = current_direction
        #             best_vertex = current_vertex
        #             best_time = current_time
        #             accepted = True
        #         elif delta_logl > 9:  # 90% confidence improvement threshold for 2 dof
        #             best_logl = current_logl
        #             best_direction = current_direction
        #             best_vertex = current_vertex
        #             best_time = current_time
        #             accepted = True
        #             print(f"Accepted update: logl improved by {delta_logl:.3f}")

        #         else:
        #             print(f"Rejected update: logl improved by {delta_logl:.3f}")
        #             accepted = False
        for sigma in GAUS_CONV_WIDTH:
            print(f"\nStarting fit with Gaussian convolution width: {sigma} ns")
            neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track, sigma=sigma)
            fit_llh = get_fitter(neg_llh, use_multiple_vertex_seeds=True, prescan_time=True)#, rtol=1e-16, atol=1e-14)
            fit_llh_jit = jax.jit(fit_llh)
            
            if best_logl is not None:
                seed_src = best_direction
                seed_pos = best_vertex
                seed_time = best_time
            else:
                seed_src = track_src
                seed_pos = centered_track_pos
                seed_time = centered_track_time
            
            solution = fit_llh_jit(seed_src, seed_pos, seed_time, fitting_event_data)
            current_logl, current_direction, current_vertex, current_time = solution

            if best_logl is None:
                best_logl = current_logl
                best_direction = current_direction
                best_vertex = current_vertex
                best_time = current_time
                accepted = True
                print(f"Initial fit: logl {current_logl:.3f}")
                continue  
            else:
                delta_logl = -2*(current_logl - best_logl)
                print(f" fit: logl {current_logl:.3f}")
                best_logl = current_logl
                best_direction = current_direction
                best_vertex = current_vertex
                best_time = current_time
                accepted = True
                print(f"logl {current_logl:.3f} previous {best_logl:.3f}")
                accepted = False
        true_src_deg = [true_src_deg[0], true_src_deg[1]]
        linefit_best_direction = np.rad2deg(best_direction)
        linefit_best_direction =[linefit_best_direction[0], linefit_best_direction[1]]
        linefit_seed_direction = np.rad2deg(seed_src)
        linefit_seed_direction =[linefit_seed_direction[0], linefit_seed_direction[1]]
        spline_src_deg = np.rad2deg(spline_src)
        spline_src_deg =[spline_src_deg[0], spline_src_deg[1]]
        best_direction_deg = np.rad2deg(best_direction)
        spline_src_deg = np.rad2deg(spline_src)
        linefit_ang_err = angular_separation_deg(
            true_src_deg[0], true_src_deg[1],
            best_direction_deg[0], best_direction_deg[1]
        )
        splinempe_ang_err = angular_separation_deg(
            true_src_deg[0], true_src_deg[1],
            spline_src_deg[0], spline_src_deg[1]
        )
        
        print(f"Reconstructed vs True Angular Distance error {linefit_ang_err:.2f} deg")
        print(f"SplineMPE vs True Angular Distance error {splinempe_ang_err:.2f} deg")

        writer.writerow([
            i,
            meta['neutrino_energy'],
            true_src_deg,
            linefit_best_direction,
            linefit_seed_direction,
            linefit_ang_err,
            spline_src_deg,
            splinempe_ang_err
        ])
        f.flush()
        elapsed = time.time() - start
        print(f"Time elapsed: {elapsed:.2f}s")
        print("--------------------------------------------------")
