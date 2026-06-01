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
from astropy.coordinates import SkyCoord
import astropy.units as u



def get_event_data(event_index: int, events_meta: pd.DataFrame, events_data: pd.DataFrame) -> pd.DataFrame:
    ev_idx = event_index
    event_meta = events_meta.iloc[ev_idx]
    event_id = event_meta['event_id']
    
    event_data = events_data[events_data['event_id'] == event_id].copy()
    return event_meta, event_data
def get_per_dom_summary_from_sim_data(
    meta: pd.DataFrame,
    pulses: pd.DataFrame,
    geo: pd.DataFrame,
    charge_key='charge',
    correct_charge=False) -> pd.DataFrame:

    df_qtot = pulses[['sensor_id', charge_key]].groupby(by=['sensor_id'], as_index=False).sum()
    df_tmin = pulses[['sensor_id', 'time']].groupby(by=['sensor_id'], as_index=False).min()
    df = df_qtot.merge(geo.iloc[df_qtot['sensor_id']], on='sensor_id', how='outer')
    df['time'] = df_tmin['time'].values

    if correct_charge == True:
        df_corr = pulses[['sensor_id', 'charge_correction']].groupby(by=['sensor_id'], as_index=False).mean()
        df['charge'] = df['charge'].values * df_corr['charge_correction'].values

    if charge_key != 'charge':
        df.rename({charge_key: 'charge'}, inplace=True, axis='columns')
    return df
def replace_early_pulse(summary_data, pulses):
    corrected_time = np.zeros(len(summary_data))
    for i, row in summary_data.iterrows():
        s_id = row['sensor_id']
        q_tot = row['charge']
        t1 = row['time']

        idx = pulses['sensor_id'] == s_id
        pulses_this_dom = pulses[idx]
        corrected_time[i] = get_first_regular_pulse(pulses_this_dom, t1, q_tot)



    summary_data['time'] = corrected_time
    return summary_data

def get_first_regular_pulse(pulses, t1, q_tot, crit_delta=10, crit_ratio = 5.e-3, crit_charge=100.):
    # technically, if we do remove early pulses, one could correct the total charge.
    # in practice, this would be an epsilon correction. Not worth adding the extra code complexity.
    # calculate ratio of charge within 10ns and 75ns of hit.
    if q_tot < crit_charge:
        return t1

    n = len(pulses)
    charge = pulses['charge'].to_numpy()
    time = pulses['time'].to_numpy()
    crit_delta_long = 75

    j = 0 # pts to end of crit_delta interval
    k = 0 # pts to end of crit_delta_long interval
    q_veto = 0
    q_long = 0
    for i in range(0, n):
        crit_time = time[i] + crit_delta
        if j < i:
            j = i

        # extend window
        while j < n and time[j] < crit_time:
            q_veto += charge[j]
            j += 1

        crit_time = time[i] + crit_delta_long
        if k < i:
            k = i

        # extend window
        while k < n and time[k] < crit_time:
            q_long += charge[k]
            k += 1

        r_veto = q_veto / q_long
        if r_veto > crit_ratio:
            # found a reasonable pulse
            # break
            break

        # remove early pulse
        q_long -= charge[i]
        q_veto -= charge[i]

    return time[i]

def plot_event(df, index, geo=None, outfile=None, plot_pdf=None):
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('pos.x [m]', fontsize=16, labelpad=-25)
    ax.set_ylabel('pos.y [m]', fontsize=16, labelpad=-25)
    ax.set_zlabel('pos.z [m]', fontsize=16, labelpad=-25)

    try:
        im = ax.scatter(geo['x'], geo['y'], geo['z'], s=0.9, c='0.7', alpha=0.8)
    except:
        pass

    im = ax.scatter(df['x'], df['y'], df['z'], s=np.sqrt(df['charge']*100), c=df['time'],
                    cmap='rainbow_r',  edgecolors='k', zorder=1000)
    ax.view_init(elev=0, azim=90)
    ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)
    cb = plt.colorbar(im, orientation="vertical", pad=0.1)
    cb.set_label(label='time [ns]', size='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.title(f'Event {index}')
    
    if plot_pdf:
        plt.close()
    else:
        plt.show()

def direction_to_zenith_azimuth(dx, dy, dz):
    """Convert direction vector to (zenith, azimuth) angles (IceCube convention).
    
    zenith: angle from downward direction (0 = down, π/2 = horizontal, π = up)
    azimuth: angle in xy-plane (0 = north, π/2 = west, π = south, 3π/2 = east)
    """
    norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8  # FIXED: was dx2 + dy2
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    zenith = np.arccos(-dz)  # IceCube convention
    azimuth = (np.arctan2(dy, dx) + np.pi) % (2 * np.pi)
    return np.array([zenith, azimuth])

def linefit_3d_time_np_weighted(fitting_event_data: pd.DataFrame, weight_by_charge=True, 
                                 sort_by_time=True, verbose=False):
    """Weighted linefit that can use hit charge as quality weight.
    
    Args:
        fitting_event_data: DataFrame with columns [sensor_id, charge, x, y, z, time]
        weight_by_charge: If True, weights by charge (hits with more charge are more reliable)
        sort_by_time: If True, sorts by time before fitting
        verbose: Print diagnostic information
    
    Returns:
        r0, t0, v, direction: Same as linefit_3d_time_np
    """
    
    data = fitting_event_data.copy()
    
    if sort_by_time:
        data = data.sort_values('time').reset_index(drop=True)
    
    positions = data.values[:, 2:5]
    times = data.values[:, 5]
    charges = data.values[:, 1]
    
    # Get weights
    if weight_by_charge:
        weights = np.maximum(charges, 0.6)  # Don't allow zero weight
        weights = weights / np.sum(weights)  # Normalize
        if verbose:
            print(f"Using charge-weighted fit (mean charge: {np.mean(charges):.3f})")
    else:
        weights = np.ones_like(times) / len(times)
        if verbose:
            print("Using unweighted fit")
    
    if verbose:
        print(f"Number of hits: {len(data)}")
        print(f"Time range: {times.min():.1f} to {times.max():.1f}")
    
    # Weighted mean-centering
    mean_pos = np.average(positions, axis=0, weights=weights)
    mean_time = np.average(times, weights=weights)
    
    delta_pos = positions - mean_pos
    delta_time = times - mean_time
    
    # Weighted least squares
    W = np.diag(weights)
    numerator = np.dot(delta_pos.T, W @ delta_time)
    denominator = np.dot(delta_time, W @ delta_time) + 1e-8
    v = numerator / denominator
    
    t0 = mean_time
    r0 = mean_pos
    
    direction = direction_to_zenith_azimuth(*v)
    
    t0 = jnp.asarray(t0)
    r0 = np.asarray(r0, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    direction = jnp.asarray(direction)
    
    return r0, t0, v, direction


def linefit_3d_time_np(fitting_event_data: pd.DataFrame):
    positions = fitting_event_data.values[:, 2:5]
    times = fitting_event_data.values[:, 5]

    """
    Perform a linefit in 3D space with time (NumPy version).
    Args:
        positions: Array of shape (N, 3) with (x, y, z) positions.
        times: Array of shape (N) with corresponding time values.
    Returns:
        r0: Fitted position at t0 (mean time).
        v: Velocity vector (direction and speed).
        direction: (zenith, azimuth)
    """

    # Mean-center the data
    mean_pos = np.mean(positions, axis=0)
    mean_time = np.mean(times)

    # Subtract means
    delta_pos = positions - mean_pos
    delta_time = times - mean_time

    # Least squares: v = (delta_pos^T delta_time) / (delta_time^T delta_time)
    numerator = np.dot(delta_pos.T, delta_time)          # shape (3,)
    denominator = np.dot(delta_time, delta_time) + 1e-8  # scalar
    v = numerator / denominator

    t0 = mean_time
    r0 = mean_pos
    

    def direction_to_zenith_azimuth(dx, dy, dz):
        """Convert a direction vector to (zenith, azimuth) angles."""
        norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        dx, dy, dz = dx / norm, dy / norm, dz / norm
        zenith = np.arccos(-dz)   # IceCube convention
        azimuth = (np.arctan2(dy, dx) + np.pi) % (2 * np.pi)
        return np.array([zenith, azimuth])

    direction = direction_to_zenith_azimuth(*v)
    t0 = jnp.asarray(t0)
    r0 = np.asarray(r0, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    direction = jnp.asarray(direction)

    return r0, t0, v, direction

def linefit(fitting_event_data: pd.DataFrame):
    """
    Perform a charge-weighted linefit in 3D space with time (NumPy version).
    Args:
        fitting_event_data: DataFrame with columns [sensor_id, charge, x, y, z, time]
    Returns:
        r0: Fitted position at t0 (mean time).
        t0: Mean (charge-weighted) time.
        v: Velocity vector (direction and speed).
        direction: (zenith, azimuth) in radians.
    """
    positions = fitting_event_data.values[:, 2:5].astype(np.float64)  # (x, y, z)
    times     = fitting_event_data.values[:, 5].astype(np.float64)    # time
    charges   = fitting_event_data.values[:, 1].astype(np.float64)    # charge weights

    # Normalize weights so they sum to 1
    w = charges / (charges.sum() + 1e-8)                              # shape (N,)
    
    # Charge-weighted means
    mean_pos  = np.average(positions, axis=0, weights=w)              # shape (3,)
    mean_time = np.average(times, weights=w)                          # scalar

    # Subtract weighted means
    delta_pos  = positions - mean_pos                                 # (N, 3)
    delta_time = times - mean_time                                    # (N,)

    # Weighted least squares: v = (w * delta_pos)^T delta_time / (w * delta_time^T delta_time)
    numerator   = np.dot((w[:, None] * delta_pos).T, delta_time)   # w[:, None] -> (N,1) broadcasts correctly
    denominator = np.dot(w * delta_time, delta_time) + 1e-8
    v  = numerator / denominator
    t0 = mean_time
    r0 = mean_pos

    def direction_to_zenith_azimuth(dx, dy, dz):
        """Convert a direction vector to (zenith, azimuth) angles (IceCube convention)."""
        norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        dx, dy, dz = dx / norm, dy / norm, dz / norm
        zenith  = np.arccos(-dz)
        azimuth = (np.arctan2(dy, dx) + np.pi) % (2 * np.pi)
        return np.array([zenith, azimuth])

    direction = direction_to_zenith_azimuth(*v)

    t0        = jnp.float64(t0)
    r0        = np.asarray(r0,        dtype=np.float64)
    v         = np.asarray(v,         dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    return r0, t0, v, direction

def get_multiple_linefit_seeds(event_data, n_seeds=5):
    """Generate multiple seed candidates for low-charge events."""
    seeds = []
    
    # Standard weighted fit
    r0, t0, v, direction = linefit_3d_time_np_weighted(event_data, weight_by_charge=True)
    seeds.append(('weighted', r0, t0, direction))
    
    # Unweighted fit
    r0, t0, v, direction = linefit_3d_time_np_weighted(event_data, weight_by_charge=False)
    seeds.append(('unweighted', r0, t0, direction))
    
    # High-charge-only fit (use top N% of hits)
    if len(event_data) > 3:
        charge_threshold = np.percentile(event_data['charge'], 75)
        high_charge_data = event_data[event_data['charge'] >= charge_threshold]
        if len(high_charge_data) > 2:
            r0, t0, v, direction = linefit_3d_time_np_weighted(high_charge_data, weight_by_charge=True)
            seeds.append(('high_charge_only', r0, t0, direction))
    
    # Early hits only (first 50% in time)
    if len(event_data) > 2:
        time_threshold = np.percentile(event_data['time'], 50)
        early_data = event_data[event_data['time'] <= time_threshold]
        if len(early_data) > 2:
            r0, t0, v, direction = linefit_3d_time_np_weighted(early_data, weight_by_charge=True)
            seeds.append(('early_hits', r0, t0, direction))
    
    return seeds


def pick_best_seed(seeds, neg_llh, fitting_event_data, threshold_charge=30):
    """Quick likelihood evaluation to pick best seed."""
    best_seed = seeds[0]  # fallback
    best_llh = np.inf
    
    for name, pos, time, direction in seeds:
        try:
            llh = neg_llh(direction, pos, time, fitting_event_data)
            if llh < best_llh:
                best_llh = llh
                best_seed = (name, pos, time, direction)
        except:
            continue
    
    return best_seed

def zenith_to_declination(zenith_deg):
    """
    Convert zenith (polar angle from +z, degrees) → declination.
      dec = 90° - zenith
    Valid zenith range: [0°, 180°] → dec range: [+90°, -90°]
    Values outside [0°,180°] indicate upstream wrapping issues in the data.
    We wrap zenith into [0°,180°] before converting.
    """
    zen = np.asarray(zenith_deg, dtype=float)
    # Wrap into [0°, 360°] then fold upper half back
    zen = zen % 360.0
    zen = np.where(zen > 180.0, 360.0 - zen, zen)
    return 90.0 - zen

def angular_sep_deg(az1, zen1, az2, zen2):
    """Great-circle separation. zen = polar angle from +z (degrees)."""
    lat1 = zenith_to_declination(zen1)
    lat2 = zenith_to_declination(zen2)
    # Wrap azimuth into [0°, 360°]
    az1 = np.asarray(az1) % 360.0
    az2 = np.asarray(az2) % 360.0
    
    c1 = SkyCoord(az1 * u.deg, lat1 * u.deg, frame="icrs")
    c2 = SkyCoord(az2 * u.deg, lat2 * u.deg, frame="icrs")
    return c1.separation(c2).deg


def plot_event_new(
    df,
    index,
    geo=None,
    outfile=None,
    plot_pdf=None,
    seed_pos=None,
    seed_time=None,
    seed_direction=None,
    track_length=500,
    azims=(90, 0),   # two azimuth viewing angles
    elev=0
):
    """
    Plot event with detector geometry, hits, and optionally the seed track.
    Creates two 3D subplots with different azimuth viewing angles.

    Args:
        df: DataFrame with hit data [x, y, z, charge, time]
        index: Event index for title
        geo: DataFrame with detector geometry [x, y, z]
        outfile: Output file path
        plot_pdf: PDF object to save figure to
        seed_pos: (3,) array with seed vertex position [x, y, z]
        seed_time: Scalar seed time
        seed_direction: (2,) array with [zenith, azimuth] in radians
        track_length: Length of track line to draw in meters
        azims: Tuple of azimuth angles for the two subplots
        elev: Elevation viewing angle
    """

    fig = plt.figure(figsize=(12, 8))

    axes = [
        fig.add_subplot(1, 2, 1, projection='3d'),
        fig.add_subplot(1, 2, 2, projection='3d')
    ]

    for i, ax in enumerate(axes):

        ax.set_xlabel('pos.x [m]', fontsize=16, labelpad=-25)
        ax.set_ylabel('pos.y [m]', fontsize=16, labelpad=-25)
        ax.set_zlabel('pos.z [m]', fontsize=16, labelpad=-25)

        # Plot detector geometry
        try:
            ax.scatter(
                geo['x'],
                geo['y'],
                geo['z'],
                s=0.9,
                c='0.7',
                alpha=0.8
            )
        except:
            pass

        # Plot hits
        im = ax.scatter(
            df['x'],
            df['y'],
            df['z'],
            s=np.sqrt(df['charge'] * 100),
            c=df['time'],
            cmap='rainbow_r',
            edgecolors='k',
            zorder=1000
        )

        # Plot seed track if provided
        if seed_pos is not None and seed_direction is not None:

            zenith, azimuth = float(seed_direction[0]), float(seed_direction[1])

            dx = np.sin(zenith) * np.cos(azimuth)
            dy = np.sin(zenith) * np.sin(azimuth)
            dz = np.cos(zenith)

            norm = np.sqrt(dx**2 + dy**2 + dz**2)

            if norm > 0:
                dx, dy, dz = dx / norm, dy / norm, dz / norm

            t_vals = np.linspace(-track_length / 2, track_length / 2, 100)

            track_x = seed_pos[0] + dx * t_vals
            track_y = seed_pos[1] + dy * t_vals
            track_z = seed_pos[2] + dz * t_vals

            ax.plot(
                track_x,
                track_y,
                track_z,
                '-',
                color='blue',
                linewidth=3,
                label='Seed Track',
                zorder=500,
                alpha=0.8
            )

            ax.scatter(
                *seed_pos,
                s=400,
                c='blue',
                marker='*',
                # edgecolors='darkred',
                linewidth=2,
                zorder=1001,
                label='Seed Vertex'
            )

        # Set different viewing angle
        ax.view_init(elev=elev, azim=azims[i])

        ax.tick_params(
            axis='both',
            which='both',
            width=1.5,
            colors='0.0',
            labelsize=16
        )

        ax.set_title(f'Event {index} | azim={azims[i]}')

        if geo is not None:
            ax.set_xlim(np.min(geo['x']), np.max(geo['x']))
            ax.set_ylim(np.min(geo['y']), np.max(geo['y']))
            ax.set_zlim(np.min(geo['z']), np.max(geo['z']))

        if seed_pos is not None:
            plt.legend(loc='upper left', fontsize=12)

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')

    if plot_pdf is not None:
        plot_pdf.savefig(fig)

    plt.show()

def angular_separation_deg(z1_deg, a1_deg, z2_deg, a2_deg):

    z1 = np.radians(z1_deg)
    a1 = np.radians(a1_deg)

    z2 = np.radians(z2_deg)
    a2 = np.radians(a2_deg)

    cos_sep = (
        np.cos(z1) * np.cos(z2)
        + np.sin(z1) * np.sin(z2) * np.cos(a1 - a2)
    )

    cos_sep = np.clip(cos_sep, -1, 1)

    return np.degrees(np.arccos(cos_sep))
