#!/usr/bin/env python

from icecube import dataio, dataclasses
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import erf
from scipy.stats import truncnorm

#from lib.geo import __theta_cherenkov
#__theta_cherenkov_deg = np.rad2deg(__theta_cherenkov)


def get_pulse_info(frame, event_id, pulses_key = 'TWSRTHVInIcePulsesIC', correction_key = None, geo_frame = None, calibrate = False):
    """
    Generates a dictionary containing all pulses for this event.
    """
    pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses_key)
    n_pulses = 0
    n_channel = 0
    q_tot = 0.0

    data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'is_HLC':[], 'charge_correction': []}

    hlc_doms = set([])
    for omkey, om_pulses in pulses.items():
            light_scale = 1.0
            if geo_frame is not None and calibrate == True:
                cal = geo_frame['I3Calibration'].dom_cal[omkey]
                light_scale = dataclasses.mean_spe_charge(cal)
                light_scale *= cal.relative_dom_eff

            n_channel += 1

            # assign sensor index
            om_idx = omkey.om - 1
            string_idx = omkey.string - 1
            sensor_idx = string_idx * 60 + om_idx

            # deal with possibility of charge correction
            correction = 1.0
            if correction_key is not None and frame.Has(correction_key):
                if omkey in frame[correction_key].keys():
                    correction = frame[correction_key][omkey]

            for i, pulse in enumerate(om_pulses):
                 n_pulses += 1
                 time = pulse.time
                 charge = pulse.charge / light_scale
                 is_HLC = int(pulse.flags & dataclasses.I3RecoPulse.PulseFlags.LC)

                 if is_HLC:
                    q_tot += charge
                    if not omkey in hlc_doms:
                        hlc_doms.add(omkey)

                 # store pulse data
                 data['event_id'].append(event_id)
                 data['time'].append(time)
                 data['charge'].append(charge)
                 data['sensor_id'].append(sensor_idx)
                 data['is_HLC'].append(is_HLC)
                 data['charge_correction'].append(correction)

    summary = {'n_pulses': n_pulses, 'n_channel': n_channel, 'n_channel_HLC': len(hlc_doms), 'q_tot': q_tot}
    return data, summary


def get_pulse_info_mcpe(frame,
                        event_id,
                        mcpe_key = 'I3MCPESeriesMapWithoutNoise',
                        pulses_key = 'TWSRTHVInIcePulsesIC',
                        correction_key = None):
    """
    DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING.
    """

    mcpe_map = frame[mcpe_key]
    pmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses_key)

    n_pulses = 0
    n_channel = 0
    q_tot = 0.0

    data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'is_HLC':[], 'charge_correction': []}

    hlc_doms = set([])
    for omkey, om_hits in mcpe_map.items():
        # these are MCPE. HLC is not defined (only for actual pulses).
        # we treat every hit as HLC.
        if not omkey in pmap.keys():
            # Skip dom's that do not exist in reco pulses
            continue

        hlc_doms.add(omkey)

        n_channel += 1

        om_idx = omkey.om - 1
        string_idx = omkey.string - 1
        sensor_idx = string_idx * 60 + om_idx

        # deal with possibility of charge correction
        correction = 1.0
        if correction_key is not None:
            correction = frame[correction_key][omkey]

        # store fudged data
        for p in om_hits:
            n_pulses += 1
            charge = float(p.npe) * correction
            data['event_id'].append(event_id)
            data['time'].append(p.time)
            data['charge'].append(charge)
            data['sensor_id'].append(sensor_idx)
            data['is_HLC'].append(1)
            data['charge_correction'].append(correction)
            q_tot += charge

    summary = {'n_pulses': n_pulses, 'n_channel': n_channel, 'n_channel_HLC': len(hlc_doms), 'q_tot': q_tot}
    return data, summary


def get_pulse_info_fudged_mcpe(frame,
                            event_id,
                            pulses_key = 'TWSRTHVInIcePulsesIC',
                            mcpe_key = 'I3MCPESeriesMapWithoutNoise',
                            correction_key = None,
                            geo_frame = None,
                            calibrate = False):
    """
    Generates a dictionary containing all pulses for this event.
    The times are taken from downsampled MCPE (keyword: CompensationFactor).
    Charges are taken from original pulses.
    Only uses DOMs for which both: pulses and MCPE exist.
    DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING.
    """

    mcpe_map = frame[mcpe_key]
    pmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses_key)

    n_pulses = 0
    n_channel = 0
    q_tot = 0.0

    data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'is_HLC':[]}

    hlc_doms = set([])
    for omkey, om_hits in mcpe_map.items():
        if not omkey in pmap.keys():
            # Skip dom's that do not exist in reco pulses
            continue

        om_pulses = pmap[omkey]
        light_scale = 1.0
        cal = geo_frame['I3Calibration'].dom_cal[omkey]
        light_scale = dataclasses.mean_spe_charge(cal)
        light_scale *= cal.relative_dom_eff
        dom_q_tot = 0

        n_channel += 1

        om_idx = omkey.om - 1
        string_idx = omkey.string - 1
        sensor_idx = string_idx * 60 + om_idx

        # deal with possibility of charge correction
        correction = 1.0
        if correction_key is not None:
            correction = frame[correction_key][omkey]

        # compute total charge
        for i, pulse in enumerate(om_pulses):
            charge = pulse.charge * correction / light_scale
            dom_q_tot += charge

            is_HLC = int(pulse.flags & dataclasses.I3RecoPulse.PulseFlags.LC)

            if is_HLC:
               q_tot += charge
               if not omkey in hlc_doms:
                   hlc_doms.add(omkey)

        # resample MCPE times (assuming that a fraction falls below discriminator
        # and therefore can't contribute as first pulse)
        spe_parameters = cal.combined_spe_charge_distribution
        args = dict()
        args['e1_amp'] = spe_parameters.exp1_amp
        args['e2_amp'] = spe_parameters.exp2_amp
        args['w1'] = spe_parameters.exp1_width
        args['w2'] = spe_parameters.exp2_width
        args['g_amp'] = spe_parameters.gaus_amp
        args['g_mean'] = spe_parameters.gaus_mean
        args['g_width'] = spe_parameters.gaus_width

        Qd = 0.2325
        frac = 1. - spe_cdf(Qd, args)
        #print(frac)

        # deal with merged npe by flattening them into series with npe=1
        #flat_om_times = [p.time for p in om_hits]
        flat_om_times = []
        for p in om_hits:
            for _ in range(p.npe):
                flat_om_times.append(p.time)

        # bound size between 1 and numper of original pulses.
        size = min(len(flat_om_times), np.round(frac * len(flat_om_times)))
        size = max(size, 1.0)
        size = int(size)
        times = np.random.choice(flat_om_times, size=size, replace=False)
        times.sort()

        # uncomment to take original pulses
        #size = len(flat_om_times)
        #times = flat_om_times

        # and we need to account for fudged charge from pulses
        # such that we preserve total charge.
        avg_pulse_q = dom_q_tot / size

        # store fudged data
        for i in range(size):
            n_pulses += 1
            data['event_id'].append(event_id)
            data['time'].append(times[i])
            data['charge'].append(avg_pulse_q)
            data['sensor_id'].append(sensor_idx)
            data['is_HLC'].append(is_HLC)

    summary = {'n_pulses': n_pulses, 'n_channel': n_channel, 'n_channel_HLC': len(hlc_doms), 'q_tot': q_tot}
    return data, summary

def spe_cdf(x, args):
    e1_amp = args['e1_amp']
    e2_amp = args['e2_amp']
    w1 = args['w1']
    w2 = args['w2']
    g_amp = args['g_amp']
    g_mean = args['g_mean']
    g_width = args['g_width']

    return e1_amp * (w1 - np.exp(-x/w1)*w1) + e2_amp * (w2 - np.exp(-x/w2)*w2) + g_amp * np.sqrt(np.pi/2) * g_width * (erf(g_mean/(np.sqrt(2)*g_width))-erf((g_mean-x)/(np.sqrt(2)*g_width)))


#def asym_gaussian(diff, sigma, r):
#    t1 = np.exp(-0.5 * (diff / sigma)**2)
#    t2 = np.exp(-0.5 * (diff / (r * sigma))**2)
#    return np.where(diff > 0, t1, t2)
#
#def get_corrected_charge(dom_pos, q_exp, loss_pos, track_dir, sigma=8, r=1):
#    # dom_pos: location of single dom. shape = (3,)
#    # loss_pos: location of all losses. shape = (N_losses, 3)
#    # track_dir: direction of track in cartesian coords. shape = (3,)
#    # q_exp: vector of expected charge from each loss for this dom. shape = (N_losses,)
#    # returns correction factor for qtot for this dom
#
#    # compute vectors between dom and losses
#    # shape: N_loss, 3
#    dx = dom_pos - loss_pos
#    dx_normed = dx / np.expand_dims(np.linalg.norm(dx, axis=1), axis=1)
#
#    # angle between line from loss to dom and track direction
#    delta = np.rad2deg(np.arccos(np.clip(np.dot(dx_normed, track_dir), -1.0, 1.0)))
#    diff_angle = __theta_cherenkov_deg - delta
#    weights = asym_gaussian(diff_angle, sigma, r)
#    return np.sum(weights * q_exp) / np.sum(q_exp)
#
#def convert_spherical_to_cartesian_direction(x):
#    """
#    x = (theta, phi)
#    """
#    track_theta = x[0]
#    track_phi = x[1]
#    track_dir_x = np.sin(track_theta) * np.cos(track_phi)
#    track_dir_y = np.sin(track_theta) * np.sin(track_phi)
#    track_dir_z = np.cos(track_theta)
#    direction = np.array([track_dir_x, track_dir_y, track_dir_z])
#    return direction
#
#def get_xyz_from_zenith_azimuth(x):
#    track_dir = convert_spherical_to_cartesian_direction(x)
#    y = -1 * track_dir
#    return y
#
#def closest_distance_dom_track(dom_pos, track_pos, track_dir):
#    """
#    dom_pos: 1D jax array with 3 components [x, y, z]
#    track_pos: 1D jax array with 3 components [x, y, z]
#    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
#    """
#
#    # vector track vertex -> dom
#    v_a = dom_pos - track_pos
#    # vector: closest point on track -> dom
#    v_d = v_a - np.dot(v_a, track_dir) * track_dir
#    dist = np.linalg.norm(v_d)
#    return dist
#
#
#def get_corrected_charge(dom_pos, q_exp, loss_pos, track_pos, track_dir, sigma=1.0, r=1.0):
#    # dom_pos: location of single dom. shape = (3,)
#    # loss_pos: location of all losses. shape = (N_losses, 3)
#    # track_dir: direction of track in cartesian coords. shape = (3,)
#    # q_exp: vector of expected charge from each loss for this dom. shape = (N_losses,)
#    # returns correction factor for qtot for this dom
#
#    # compute vectors between dom and losses
#    # shape: N_loss, 3
#    dx = dom_pos - loss_pos
#    dx_normed = dx / np.expand_dims(np.linalg.norm(dx, axis=1), axis=1)
#
#    # angle between line from loss to dom and track direction
#    delta = np.arccos(np.clip(np.dot(dx_normed, track_dir), a_min=-1.0, a_max=1.0))
#
#    delta_ = np.where(delta <= np.pi/2.0, delta, np.pi-delta)
#
#    dist = closest_distance_dom_track(dom_pos, track_pos, track_dir)
#    yc = dist / np.tan(__theta_cherenkov)
#    yl = dist / np.tan(delta_)
#
#    dy = np.where(delta < __theta_cherenkov, yl - yc, -(yc - yl))
#    dy = np.where(delta < np.pi/2, dy, -(yl + yc))
#    dy = np.where(delta != np.pi/2, dy, -yc)
#
#    dx = dy / dist
#
#    weights = asym_gaussian(dx, sigma, r)
#    return np.sum(weights * q_exp) / np.sum(q_exp)
#
#
#def get_pulse_info_w_qtot_correction(frame,
#                                     geo_frame,
#                                     event_id,
#                                     pulses_key = 'TWSRTHVInIcePulsesIC',
#                                     millipede_qexp_key = 'MCMostEnergeticTrack_I3MCTree_ExQ',
#                                     track_key = 'MCMostEnergeticTrack'):
#    """
#    Generates a dictionary containing all pulses for this event.
#    And scales down charge for DOMs that are dominated by off-time stochastic losses.
#    """
#
#    geo = geo_frame['I3Geometry'].omgeo
#    pmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses_key)
#    expq_dict = frame[millipede_qexp_key]
#
#    I3MCTree = frame['I3MCTree']
#    loss_pos = []
#    for p in I3MCTree.get_daughters(frame['MCPrimary1']):
#         for loss in I3MCTree.get_daughters(p):
#               if not 'Mu' in str(loss.type):
#                    pos = loss.pos
#                    loss_pos.append([pos.x, pos.y, pos.z])
#
#    track = frame[track_key]
#    track_pos = np.array([track.pos.x, track.pos.y, track.pos.z])
#    track_dir_xyz = get_xyz_from_zenith_azimuth([track.dir.zenith, track.dir.azimuth])
#
#    n_pulses = 0
#    n_channel = 0
#    q_tot = 0.0
#
#    sigma_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#    data = {'event_id': [], 'sensor_id': [], 'time': [], 'charge': [], 'corrected_charge':[], 'is_HLC':[]}
#    for sigma in sigma_list:
#        data[f'corrected_charge_{sigma:.1f}'] = []
#
#    hlc_doms = set([])
#    for omkey, om_pulses in pmap.items():
#            n_channel += 1
#
#            # assign sensor index
#            om_idx = omkey.om - 1
#            string_idx = omkey.string - 1
#            sensor_idx = string_idx * 60 + om_idx
#
#            # compute charge correction factor
#            pos = geo[omkey].position
#            dom_pos = np.array([pos.x, pos.y, pos.z])
#            q_exp = np.array(expq_dict[omkey])
#
#            corr_facs = []
#            for sigma in sigma_list:
#                correction_factor = get_corrected_charge(dom_pos, q_exp, loss_pos, track_pos, track_dir_xyz, sigma=sigma, r=1.0)
#                corr_facs.append(correction_factor)
#
#            for i, pulse in enumerate(om_pulses):
#                 n_pulses += 1
#                 time = pulse.time
#                 charge = pulse.charge
#                 is_HLC = int(pulse.flags & dataclasses.I3RecoPulse.PulseFlags.LC)
#                 if is_HLC:
#                    q_tot += charge
#                    if not omkey in hlc_doms:
#                        hlc_doms.add(omkey)
#
#                 # store pulse data
#                 data['event_id'].append(event_id)
#                 data['time'].append(time)
#                 data['charge'].append(charge)
#
#                 for correction_factor, sigma in zip(corr_facs, sigma_list):
#                    corrected_charge = charge * correction_factor
#                    data[f'corrected_charge_{sigma:.1f}'].append(corrected_charge)
#                 data['sensor_id'].append(sensor_idx)
#                 data['is_HLC'].append(is_HLC)
#
#    summary = {'n_pulses': n_pulses, 'n_channel': n_channel, 'n_channel_HLC': len(hlc_doms), 'q_tot': q_tot}
#    return data, summary
