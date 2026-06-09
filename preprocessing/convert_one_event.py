from icecube import dataio, hdfwriter, icetray, MuonGun, dataclasses
from icecube.sim_services.label_events import MCLabeler, MuonLabels
from icecube.sim_services.label_events import ClassificationConverter
from icecube.icetray import I3Tray
from icecube.dataclasses import I3Particle
from icecube.icetray import I3Units
from icecube.icetray import *
from icecube.sim_services.label_events.enums import classification
import numpy as np
import warnings

import numpy as np
from scipy.stats import norm
from scipy.special import erf
from scipy.stats import truncnorm
import pandas as pd
import os
import glob
# os.path.append('/mnt/home/baburish/jax/TriplePandelReco_JAX')
from lib.pulse_extraction_from_i3 import get_pulse_info
from lib.muon_energy import add_muon_energy


muon_label_list = []
def muonframe(frame):
  global muon_label_list
  try:
    seed = frame['coincident_muons'].value
    print("Seed=", seed)
    if seed > 0:
    # if frame['coincident_muons'] == True:
      print("Coincident muons! Skip!")
      False
    else:
      print("No coincident muons. Keep!")
      True
      # print(seed)
      muon_label_list.append(seed)
  except:
    print("Error")

RECOMPUTE_MU_E=True
def framestats(frame):
  global muon_label_list
  global event_header
  global interaction_type
  global most_energetic_track
  global primary_neutrino
  global spline_mpe
  global muon_energy_at_interaction
  global muon_energy_at_det
  global muon_energy_leaving
  global meta_frames
  global pulse_frames
  global event_count
  global pulse_data 
  global meta_data


  # print("Processing event", frame['I3EventHeader'])
  # if RECOMPUTE_MU_E:
  #           # Compute true properties of muon.
  #           #print("recomputing muon energy.")
  #           add_muon_energy(frame)
  try:
      muon_energy_at_interaction = frame[meta_keys['mc_muon_energy_at_interaction']].value # I3Double
      muon_energy_at_det =  frame[meta_keys['mc_muon_energy_at_detector_entry']].value # I3Double
      muon_energy_leaving = frame[meta_keys['mc_muon_energy_at_detector_leave']].value # I3Double
      print("Muon energy at interaction=", muon_energy_at_interaction)
  except:
      print("Missing a key. Skip!")
  
  # most_energetic_track = frame[meta_keys['mc_most_energetic_muon']] # I3Particle
  true_ra = frame['trueRa']
  true_dec = frame['trueDec']
  # spline_mpe = frame[meta_keys['spline_mpe']]
  primary_neutrino = frame[meta_keys['mc_primary_neutrino']]
  tnf_ra = frame[meta_keys['LT-ra']]
  tnf_dec = frame[meta_keys['LT-dec']]
  print("True Dir=", true_ra, true_dec)
  # print("Spline MPE Dir=", spline_mpe.dir.zenith, spline_mpe.dir.azimuth)
  print("LT-TNF Dir=", tnf_ra, tnf_dec)
  try:
    seed = frame['coincident_muons']
    event_header = frame['I3EventHeader']
    interaction_type = frame['I3MCWeightDict']['InteractionType']
    most_energetic_track = frame[meta_keys['mc_most_energetic_muon']] # I3Particle
    primary_neutrino = frame[meta_keys['mc_primary_neutrino']] # I3Particle
    spline_mpe = frame[meta_keys['spline_mpe']]
    tnf_ra = frame[meta_keys['LT-ra']]
    tnf_dec = frame[meta_keys['LT-dec']]
    muon_label_list.append(seed)
    print("True Dir=", most_energetic_track.dir.zenith, most_energetic_track.dir.azimuth)
    print("Spline MPE Dir=", spline_mpe.dir.zenith, spline_mpe.dir.azimuth)
    print("LT-TNF Dir=", tnf_ra, tnf_dec)
    # print("Loaded")
  except:
    primary_neutrino = None
    most_energetic_track = None
    spline_mpe = None
  
    print("Error")
    # exit(1)

  is_CC_interaction = interaction_type < 1.5
#   print("Muon energy at detector:", muon_energy_at_det)
#   pass_muon_energy = np.isfinite(muon_energy_at_det) and muon_energy_at_det > min_muon_energy_at_detector and muon_energy_at_det < max_muon_energy_at_detector
  pass_muon_energy = True
  # energy_ratio = muon_energy_at_interaction  / most_energetic_track.energy
  # found_correct_muon = energy_ratio < 0.9 or energy_ratio > 0.9
  # has_sensible_muon = np.logical_and(pass_muon_energy, energy_ratio)
  if meta_keys['bkg_mc_tree'] == 'I3MCTree':
     has_no_coinc = len(frame['I3MCTree'].get_primaries()) == 1
  else:
     has_no_coinc = len(frame[meta_keys['bkg_mc_tree']]) == 0
  # has_sensible_muon = np.logical_and(has_sensible_muon, has_no_coinc)
  has_sensible_muon = True
  print(is_CC_interaction, has_sensible_muon)
  if np.logical_and(is_CC_interaction, has_sensible_muon):
    print("Retain event.")
    # Retain event.
    event_count += 1

    event_id = event_header.run_id * n_events_per_file + event_header.event_id

    # Get all pulses.
    event_pulse_data, summary = get_pulse_info(frame, event_id, pulses_key=meta_keys['pulses'])
    print("Summary=", summary)
    # print(event_id, summary)
    # Store.
    for key in pulse_data.keys():
        pulse_data[key] += event_pulse_data[key]

    # Get meta_data.
    event_last_pulse_idx = event_first_pulse_idx + summary['n_pulses'] - 1
    meta_data['event_id'].append(event_id)
    meta_data['idx_start'].append(event_first_pulse_idx)
    meta_data['idx_end'].append(event_last_pulse_idx)
    print(primary_neutrino)
    try:
      meta_data['neutrino_energy'].append(primary_neutrino.energy)
      meta_data['muon_energy'].append(muon_energy_at_interaction)
      meta_data['muon_energy_at_detector'].append(muon_energy_at_det)
    except:
      meta_data['neutrino_energy'].append(np.nan)
      meta_data['muon_energy'].append(np.nan)
      meta_data['muon_energy_at_detector'].append(np.nan)

    # if np.isfinite(muon_energy_leaving):
    #     meta_data['muon_energy_lost'].append(muon_energy_at_det - muon_energy_leaving)
    # else:
    #     # lost all energy inside the detector
    #     meta_data['muon_energy_lost'].append(muon_energy_at_det)
  try:
    meta_data['q_tot'].append(summary['q_tot'])
    meta_data['n_channel'].append(summary['n_channel'])
    meta_data['n_channel_HLC'].append(summary['n_channel_HLC'])
    meta_data['muon_zenith'].append(most_energetic_track.dir.zenith)
    meta_data['muon_azimuth'].append(most_energetic_track.dir.azimuth)
    meta_data['muon_time'].append(most_energetic_track.time)
    meta_data['muon_pos_x'].append(most_energetic_track.pos.x)
    meta_data['muon_pos_y'].append(most_energetic_track.pos.y)
    meta_data['muon_pos_z'].append(most_energetic_track.pos.z)
    meta_data['spline_mpe_zenith'].append(spline_mpe.dir.zenith)
    meta_data['spline_mpe_azimuth'].append(spline_mpe.dir.azimuth)
    meta_data['spline_mpe_time'].append(spline_mpe.time)
    meta_data['spline_mpe_pos_x'].append(spline_mpe.pos.x)
    meta_data['spline_mpe_pos_y'].append(spline_mpe.pos.y)
    meta_data['spline_mpe_pos_z'].append(spline_mpe.pos.z)
  except:
    meta_data['q_tot'].append(np.nan)
    meta_data['n_channel'].append(np.nan)
    meta_data['n_channel_HLC'].append(np.nan)
    meta_data['muon_zenith'].append(np.nan)
    meta_data['muon_azimuth'].append(np.nan)
    meta_data['muon_time'].append(np.nan)
    meta_data['muon_pos_x'].append(np.nan)
    meta_data['muon_pos_y'].append(np.nan)
    meta_data['muon_pos_z'].append(np.nan)
    meta_data['spline_mpe_zenith'].append(np.nan)
    meta_data['spline_mpe_azimuth'].append(np.nan)
    meta_data['spline_mpe_time'].append(np.nan)
    meta_data['spline_mpe_pos_x'].append(np.nan)
    meta_data['spline_mpe_pos_y'].append(np.nan)
    meta_data['spline_mpe_pos_z'].append(np.nan)
    
  else:
    print("Skip event.")
    # Skip event.

def label_muons(file):
    i3file = file
    gcd = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz"
    tray = I3Tray()
    tray.Add("I3Reader", Filenamelist=[gcd, i3file])
    tray.Add(
    MCLabeler,
    event_properties_name=None,
    mctree_name='I3MCTree_preMuonProp',
    weight_dict_name='I3MCWeightDict',
    bg_mctree_name="I3MCTree_preMuonProp",
    )
    tray.Add(muonframe, Streams=[icetray.I3Frame.Physics])
    tray.Add(framestats, Streams=[icetray.I3Frame.Physics])
    tray.Execute()
    # tray.PrintUsage()
    return meta_data, pulse_data, event_count

# dataset_id = "22853"
dataset_id = "22646"
file_index_start = 0
file_index_end = 10
outdir = "./test_output"
os.makedirs(outdir, exist_ok=True)

# directory = "/mnt/research/IceCube/Gupta-Reco/l322645/0000000-0000999"
# pattern = "FinalLevel_NuMu_NuGenCCNC.022853.000354.i3.zst"
# directory ="/mnt/research/IceCube/Gupta-Reco/22644/0000000-0000999"
# pattern = "FinalLevel_NuMu_NuGenCCNC.022644.000969.i3.zst"
directory ="/mnt/scratch/harnisc6/LightningTracks/07_final_cut_models/nugen/baseline_numu_22644/0000000-0000999"
pattern = "Level2_NuMu_NuGenCCNC.022644.000499.i3.zst"
file_pattern = os.path.join(directory, pattern)

i3files = sorted(glob.glob(file_pattern))
pulse_frames = []
meta_frames = []

total_event_count = 0
for i, i3file in enumerate(i3files):
    n_events_per_file = int(1.e5)
    event_count = 0
    event_first_pulse_idx = 0 # inclusive
    event_last_pulse_idx = 0 # inclusive
    meta_keys = dict()
    # meta_keys['pulses'] = 'TWSRTHVInIcePulsesIC'
    meta_keys['pulses'] = 'SplitInIcePulses'
    # meta_keys['mc_primary_neutrino'] = 'MCPrimary1'
    meta_keys['mc_primary_neutrino'] = 'I3MCWeightDict'
    meta_keys['mc_most_energetic_muon'] = 'MCMostEnergeticTrack'
    meta_keys['spline_mpe'] = 'SplineMPEIC'
    meta_keys['LT-ra'] = 'LT_TNF_ra'
    meta_keys['LT-dec'] = 'LT_TNF_dec'
    meta_keys['mc_muon_energy_at_interaction'] = 'TrueMuonEnergyAtInteraction'
    meta_keys['mc_muon_energy_at_detector_entry']  = 'TrueMuoneEnergyAtDetectorEntry'
    meta_keys['mc_muon_energy_at_detector_leave'] = 'TrueMuoneEnergyAtDetectorLeave'
    meta_keys['bkg_mc_tree'] = 'I3MCTree'
    pulse_data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'is_HLC':[]}

    meta_data = {'event_id': [], 'idx_start': [], 'idx_end': [], 'n_channel_HLC': []}
    meta_data.update({'neutrino_energy': [], 'muon_energy': [], 'muon_energy_at_detector': []})
    meta_data.update({'muon_energy_lost': [], 'q_tot': [], 'n_channel': []})
    meta_data.update({'muon_zenith': [], 'muon_azimuth': [], 'muon_time': []})
    meta_data.update({'muon_pos_x': [], 'muon_pos_y': [], 'muon_pos_z': []})
    meta_data.update({'spline_mpe_zenith': [], 'spline_mpe_azimuth': [], 'spline_mpe_time': []})
    meta_data.update({'spline_mpe_pos_x': [], 'spline_mpe_pos_y': [], 'spline_mpe_pos_z': []})
    min_muon_energy_at_detector = 100 # GeV
    max_muon_energy_at_detector = 1000000 # GeV
    print(f"Processing file {i+1}/{len(i3files)}: {i3file}")
    meta_data2, pulse_data2, event_count = label_muons(i3file)
    df_pulses = pd.DataFrame.from_dict(pulse_data2)
    df_meta = pd.DataFrame.from_dict(meta_data2)
    # print(df_pulses.head())
    pulse_frames.append(df_pulses)
    meta_frames.append(df_meta)
    print("Length of pulses", len(df_pulses))
    print("Len of pulse frames", len(pulse_frames))
    print("Length of meta frames", len(meta_frames))
    print("Event counts=", event_count)
    total_event_count += event_count

df_pulses2 = pd.concat(pulse_frames).reset_index(drop=True)
df_meta2 = pd.concat(meta_frames).reset_index(drop=True)
df_pulses2 = df_pulses2.drop_duplicates(subset=['event_id', 'sensor_id', 'time'])
df_meta2 = df_meta2.drop_duplicates(subset=['event_id'])


# for i, i3file in enumerate(i3files):
# # for i in range(0, 3):
#     i3file = i3files[i]
#     print(f"Processing file {i+1}/{len(i3files)}: {i3file}")
#     meta_data2, pulse_data2, event_count = label_muons(i3file)
#     df_pulses = pd.DataFrame.from_dict(pulse_data2)
#     df_meta = pd.DataFrame.from_dict(meta_data2)
#     pulse_frames.append(df_pulses)
#     meta_frames.append(df_meta)
#     print("Len of pframes", len(pulse_frames))
#     print("Length of meta frames", len(meta_frames))
#     print("Event counts=", event_count)
# print("No of pulse frames", len(pulse_frames))
# df_pulses2 = pd.concat(pulse_frames).reset_index(drop=True)
# df_meta2 = pd.concat(meta_frames).reset_index(drop=True)

# ofile_pulses = os.path.join(outdir, f"pulses_ds_{dataset_id}_from_{file_index_start}_to_{file_index_end}_10_to_100TeV.ftr")
# ofile_meta   = os.path.join(outdir, f"meta_ds_{dataset_id}_from_{file_index_start}_to_{file_index_end}_10_to_100TeV.ftr")

# df_pulses2.to_feather(ofile_pulses, compression='zstd')
# df_meta2.to_feather(ofile_meta, compression='zstd')
# print(f"Stored {event_count} events in outfiles:\n  {ofile_pulses}\n  {ofile_meta}")