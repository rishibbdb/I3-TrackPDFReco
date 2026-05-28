import numpy as np
import jax
import jax.numpy as jnp
from lib.geo import convert_spherical_to_cartesian_direction

def remove_early_pulses(eval_network_doms_and_track, data, track_pos, track_dir, track_time):
    crit = -50.0 # ns
    #crit = -30
    _, _, _, geo_times = eval_network_doms_and_track(data[:,:3], track_pos, track_dir)
    delay_times = data[:, 3] - geo_times - track_time
    idx = delay_times > crit
    filtered_data = data[idx]
    return filtered_data


def remove_early_pulses_and_noise_candidate_doms(eval_network_doms_and_track, data, track_pos, track_dir, track_time):
    crit_time = -50 # ns
    crit_charge = 0.05 # pe
    _, _, _, geo_times, predicted_charge = eval_network_doms_and_track(data[:,:3], track_pos, track_dir)

    # filter early pulses
    delay_times = data[:, 3] - geo_times - track_time
    idx_time = delay_times > crit_time

    # filter predicted low charger doms
    charges = data[:, 4]
    predicted_charge = jnp.squeeze(jnp.sum(charges) / jnp.sum(predicted_charge) * predicted_charge)
    idx_charge = predicted_charge > crit_charge

    idx = jnp.logical_and(idx_time, idx_charge)
    filtered_data = data[idx]
    return filtered_data


def get_clean_pulses_fn(eval_network_doms_and_track_fn, n_pulses=1):

    def clean_pulses(data, mctruth):
        track_src = mctruth[2:4]
        track_time = mctruth[4]
        track_pos = mctruth[5:8]

        crit = -50.0 # ns
        #_, _, _, geo_times, _ = eval_network_doms_and_track_fn(data[:,:3], track_pos, track_src)
        _, _, _, geo_times = eval_network_doms_and_track_fn(data[:,:3], track_pos, track_src)
        delay_times = data[:, 3] - geo_times - track_time
        idx = delay_times > crit
        idx = idx.reshape((data.shape[0], 1))
        data_clean = jnp.where(idx, data, jnp.zeros((1,3+2*n_pulses)))
        return data_clean

    return clean_pulses


def get_clean_pulses_fn_v(eval_network_doms_and_track_fn, n_pulses=1):
    clean_pulses = get_clean_pulses_fn(eval_network_doms_and_track_fn, n_pulses=n_pulses)
    clean_pulses_v = jax.jit(jax.vmap(clean_pulses, 0, 0))
    return clean_pulses_v


def get_prop_perp_direcs(theta, phi):
        """[given a direction in the sky it returns three orthonormal three dimensional vectors,
            two perpendicular to the initial direction and one along it]

        Args:
            theta ([float]): Theta angle
            phi ([float]): Phi angle

        Returns:
            v_dir ([numpy array of floats]): the vector along the direction
            dir1 ([numpy array of floats]): one of the two vectors orthogonal to the direction
            dir2 ([numpy array of floats]): the other vector orthogonal to the direction
        """

        x = jnp.array([theta, phi])
        v_dir = convert_spherical_to_cartesian_direction(x)

        dir1 = jnp.array(
            [
                jnp.cos(phi) * jnp.sin(theta - jnp.pi / 2.0),
                jnp.sin(phi) * jnp.sin(theta - jnp.pi / 2.0),
                jnp.cos(theta - jnp.pi / 2.0),
            ]
        )

        dir2 = jnp.cross(v_dir, dir1)

        return v_dir, dir1, dir2


def get_vert_seeds(
    vert_mid, direc, v_ax=[-40.0, 40.0], r_ax=[150.0], ang_steps=3
    ):
    """[Given a vertex and a direction it returns some vertex seeds to perform a SplineMPE fit
        these vertexes are by default 7, one is the initial one, the othes 6 are chosen along
        a cylinder with characteristics specified by v_ax and r_ax and ang_steps]

    Args:
        vert_mid ([]): The initial vertex
        direc ([]): The direction in the sky
        v_ax ([list of floats], optional): the list of steps along the direction to do to find
                                           the vertexes
        r_ax ([list of floats], optional): the list of radius of the cylinders used to find the
                                           vertexes
        ang_steps([int], optional): the number of seeds to be taken on each basis of a cylinder

    Returns:
        pos_seeds ([list of I3Position]): the list of vertex seeds

    by Giacomo Sommani
    """
    theta, phi = direc[0], direc[1]
    ang_ax = np.linspace(0, 2.0 * np.pi, ang_steps + 1)[:-1]

    # Angular space batween each seed.
    dang = (ang_ax[1] - ang_ax[0]) / 2.0
    v_dir, dir1, dir2 = get_prop_perp_direcs(theta, phi)

    # In the following, the function constructs a list of seed vertexes.
    pos_seeds = [vert_mid]

    for i, vi in enumerate(v_ax):
        v = vi * v_dir

        for j, r in enumerate(r_ax):
            for ang in ang_ax:
                d1 = r * jnp.cos(ang + (i + j) * dang) * dir1
                d2 = r * jnp.sin(ang + (i + j) * dang) * dir2

                x = v[0] + d1[0] + d2[0]
                y = v[1] + d1[1] + d2[1]
                z = v[2] + d1[2] + d2[2]

                pos = jnp.array([
                    vert_mid[0] + x,
                    vert_mid[1] + y,
                    vert_mid[2] + z
                ])

                pos_seeds.append(pos)

    return pos_seeds


def get_vertex_seeds(seed, direction):
    """[Given a vertex and a direction it returns some vertex seeds to perform a SplineMPE fit
        these vertexes are by default 7, one is the initial one, the othes 6 are chosen along
        a cylinder with characteristics specified by v_ax and r_ax and ang_steps]

    Args:
        vert_mid ([]): The initial vertex
        direc ([]): The direction in the sky
        v_ax ([list of floats], optional): the list of steps along the direction to do to find
                                           the vertexes
        r_ax ([list of floats], optional): the list of radius of the cylinders used to find the
                                           vertexes
        ang_steps([int], optional): the number of seeds to be taken on each basis of a cylinder

    Returns:
        pos_seeds ([list of I3Position]): the list of vertex seeds

    """
    #v_ax = jnp.array([-30.0, -15.0, 15.0, 30.0])
    #r_ax = jnp.array([15.0, 30.0])

    v_ax = jnp.array([-20.0 , 20.0])
    r_ax = jnp.array([20.0])

    #v_ax = jnp.array([-40.0, 40.0])
    #r_ax = jnp.array([150.])
    ang_steps = 3

    theta, phi = direction[0], direction[1]

    ang_ax = jnp.linspace(0, 2.0 * jnp.pi, ang_steps + 1)[:-1]
    dang = (ang_ax[1] - ang_ax[0]) / 2.0

    r_ax_idx = jnp.arange(r_ax.shape[0])
    v_dir, dir1, dir2 = get_prop_perp_direcs(theta, phi)


    pos_dx = calc_vertex_seeds_v(r_ax_idx, r_ax, v_ax, v_dir, dir1, dir2, ang_ax)
    pos_dx = pos_dx.reshape((pos_dx.shape[0]*pos_dx.shape[1], pos_dx.shape[2]))

    seeds = seed + pos_dx
    return jnp.concatenate([seeds, seed.reshape((1,3))])


def calc_vertex_seeds(r_ax_idx, r, v_ax, v_dir, dir1, dir2, ang_ax):
    v_dir = jnp.expand_dims(v_dir, axis=0)
    v_ax = jnp.expand_dims(v_ax, axis=-1)
    v = v_ax * v_dir # (N_vax, 3) ==> N_vax * [[x, y, z]]
    v_ax_idx = jnp.arange(v.shape[0]) # N_vax
    dang = (ang_ax[1] - ang_ax[0]) / 2.0

    # (N_ang_ax, N_vax)
    d1 = r * jnp.cos(jnp.expand_dims(ang_ax, axis=-1) + (jnp.expand_dims(v_ax_idx, axis=0) + r_ax_idx) * dang)
    d2 = r * jnp.sin(jnp.expand_dims(ang_ax, axis=-1) + (jnp.expand_dims(v_ax_idx, axis=0) + r_ax_idx) * dang)

    # (N_ang_ax, N_vax, 1) * (1, 1, 3)
    d1 = jnp.expand_dims(d1, axis=-1) * jnp.expand_dims(jnp.expand_dims(dir1, axis=0), axis=0)
    d2 = jnp.expand_dims(d2, axis=-1) * jnp.expand_dims(jnp.expand_dims(dir2, axis=0), axis=0)

    # (N_ang_ax, N_vax, 3)
    x = jnp.expand_dims(v, axis=0) + d1 + d2
    # (N_ang_ax * N_vax , 3)
    x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
    return x

calc_vertex_seeds_v = jax.jit(jax.vmap(calc_vertex_seeds, (0, 0, None, None, None, None, None), 0))

'''
def get_first_regular_pulse_old(pulses, t1, q_tot, crit_delta=10, crit_ratio = 2.e-3, crit_charge=300.):
    # technically, if we do remove early pulses, one could correct the total charge.
    # in practice, this would be an epsilon correction. Not worth adding the extra code complexity.
    # calculate ratio between charge within 10ns of pulse and total charge.
    if q_tot < crit_charge:
        return t1

    n = len(pulses)
    charge = pulses['charge'].to_numpy()
    time = pulses['time'].to_numpy()

    j = 0
    q_veto = 0
    for i in range(0, n):
        crit_time = time[i] + crit_delta
        if j < i:
            j = i

        # extend window
        while j < n and time[j] < crit_time:
            q_veto += charge[j]
            j += 1

        r_veto = q_veto / q_tot
        if r_veto > crit_ratio:
            # found a reasonable pulse
            # break
            break

        # remove early pulse
        q_tot -= charge[i]
        q_veto -= charge[i]

    return time[i]
'''

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
