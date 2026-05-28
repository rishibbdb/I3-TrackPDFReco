import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np


def linefit_3d_time_jnp(fitting_event_data: pd.DataFrame):
    positions = fitting_event_data.values[:,2:5]
    times = fitting_event_data.values[:,5]
    """
    Perform a linefit in 3D space with time.
    Args:
        positions: Array of shape (N, 3) with (x, y, z) positions.
        times: Array of shape (N) with corresponding time values.
    Returns:
        r0: Fitted position at t0 (mean time).
        v: Velocity vector (direction and speed).
    """
    # Mean-center the data
    mean_pos = jnp.mean(positions, axis=0)
    mean_time = jnp.mean(times)
    # Subtract means
    delta_pos = positions - mean_pos
    delta_time = times - mean_time
    # Solve the least squares fit: v = (delta_pos^T delta_time) / (delta_time^T delta_time)
    numerator = jnp.dot(delta_pos.T, delta_time)  # shape (3,)
    denominator = jnp.dot(delta_time, delta_time) + 1e-8  # scalar, add epsilon for stability
    v = numerator / denominator
    t0 = mean_time
    r0 = mean_pos  # position at t0 = mean_time
    def direction_to_zenith_azimuth(dx, dy, dz):
        """Convert a direction vector to (zenith, azimuth) angles."""
        norm = jnp.sqrt(dx**2 + dy**2 + dz**2) + 1e-8  # avoid div-by-zero
        dx, dy, dz = dx / norm, dy / norm, dz / norm
        zenith = jnp.arccos(-dz)  # IceCube-style: zenith = 0 is downward
        azimuth = (jnp.arctan2(dy, dx) + jnp.pi) % (2 * jnp.pi)
        return jnp.array([zenith, azimuth])
    direction = direction_to_zenith_azimuth(*v)
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