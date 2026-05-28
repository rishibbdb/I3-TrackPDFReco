import jax
import jax.numpy as jnp
import optimistix as optx
from lib.experimental_methods import get_vertex_seeds

def get_fitter(neg_llh,
                use_multiple_vertex_seeds=False,
                prescan_time=False,
                scale=100.0,
                scale_rad=50.0,
                rtol=1e-10,
                atol=1e-6,
                use_batches=False):
    """Creates a fitter() function that performs a 5D likelihood optimization.
    Note: The argumenst are used globally
    within functions defined in the body.

    Args
    ----
        neg_llh: A valid negative log likelihood function.
        use_multiple_vertex_seeds: If True, the fitter will perform the
            reconstruction starting from additional vertex seeds.
            This increases robustness against local minima
            but increases the run-time.
        prescan_time: If True, the fitter will search for the best time
            for a given vertex seed.
            This can improve convergence at the cost of run-time.
        scale: Re-scales the vertex coordinate units during optimization.
            (maps step size of 1 to value of scale).
        scale_rad: Re-scales the directional coordinate units during optimization.
            (maps step size of 1 to value of scale).
        rtol: relative tolerance of the optimizer (see optimistix.BFGS)
        atol: abolsute tolerance of the optimizer (see optimistix.BFGS)
        use_batches: If True, the reconstruction is performed on batches
            of events. The fitter function that is created here will be
            modified such that all its arguments have an additional dimension
            at index 0 - the event dimension. Argument shapes are
            shape=(batch_size, argument_size), where batch_size corresponds
            to the number of events in the batch.

    Returns
    -------
        fitter: A function that performs the likelihood fit
            according with the given properties / behavior.
    """

    # Vectorize likelihood along time argument.
    # Reminder: arguments are (direction, vertex, time, data).
    neg_llh_time_v = jax.vmap(neg_llh, (None, None, 0, None), 0)

    def get_track_time(track_dir, track_vertex, seed_time, data):
        """Find time that best matches the given vertex seed
        and track direction. I.e. the time that yields the lowest
        log likelihood value for the given track parameters.

        Args
        ----
            track_dir: jnp.array([zenith, azimuth]) in radians
            track_vertex: jnp.array([x, y, z]) in m
            seed_time: jnp.array(t) in ns
            data: jnp.array(data) with shape
                (n_sensors, n_features) = (N, 5)

        Returns
        -------
            best_time: jnp.array(float)
        """
        dt = 100. # we search 100ns before and after seed_time
        nt = 20 # number of evaluation points
        time = jnp.linspace(seed_time - dt, seed_time + dt, nt)
        llh = neg_llh_time_v(track_dir, track_vertex, time, data)

        return time[jnp.argmin(llh, axis=0)]

    # Vectorize across vertex dimension. This allows performing
    # this operation for multiple vertex seeds at the
    # given track direction.
    # Reminder: arguments are (direction, vertex, time, data).
    get_track_time_v = jax.vmap(
        get_track_time,
        (None, 0, None, None),
        0
    )

    # Define the likelihood function for the 5D optimization
    def neg_llh_5D(x, args):
        # project back if outside of [0, pi] x [0, 2*pi]
        zenith = x[0] / scale_rad
        azimuth = x[1] / scale_rad
        zenith = jnp.fmod(zenith, 2.0*jnp.pi)
        zenith = jnp.where(zenith < 0, zenith+2.0*jnp.pi, zenith)
        cond = zenith > jnp.pi
        zenith = jnp.where(cond, -1.0*zenith+2.0*jnp.pi, zenith)
        azimuth = jnp.where(cond, azimuth-jnp.pi, azimuth)

        azimuth = jnp.fmod(azimuth, 2.0*jnp.pi)
        azimuth = jnp.where(azimuth < 0, azimuth+2.0*jnp.pi, azimuth)
        projected_dir = jnp.array([zenith, azimuth])

        track_time, data = args
        return neg_llh(projected_dir, x[2:]*scale, track_time, data)


    def reconstruct_event(track_vertex_seed, track_dir_seed, track_time, data):
        """Performs a single event reconstruction.

        Args
        ----
        track_vertex_seed: jnp.array([x, y, z]) in m
        track_dir_seed: jnp.array([zenith, azimuth]) in radians
        track_time: jnp.array(t) in ns
        data: jnp.array(data) with shape
                (n_sensors, n_features) = (N, 5)

        Returns
        -------
            Best-fit Negative loglikelihood value (neglogl) and
            corresponding coordinates as tuple(jnp.array, jnp.array([zenith, azimuth, x,y,z]).
        """
        solver = optx.BestSoFarMinimiser(optx.BFGS(rtol=rtol, atol=atol, use_inverse=True))
        args = (track_time, data)
        x0 = jnp.concatenate([track_dir_seed*scale_rad, track_vertex_seed/scale])
        sol = optx.minimise(neg_llh_5D,
                            solver,
                            x0,
                            args=args,
                            throw=False).value

        sol_dir = sol[:2] / scale_rad
        sol_pos = sol[2:] * scale

        return neg_llh_5D(sol, args), sol_dir, sol_pos

    # Vectorize over vertex argument
    reconstruct_event_v = jax.vmap(reconstruct_event, (0, None, None, None), 0)

    # Vectorize over vertex and time arguments
    reconstruct_event_vt = jax.vmap(reconstruct_event, (0, None, 0, None), 0)

    def run_reconstruction(track_dir_seed, vertex_seed, track_time, data):
        """Wraps a single reconstruction for a given track direction seed.
        Allows reconstructing that event multiple times with different
        track seed values. And provides possibility to adjust the corresponding
        time constant to provide best starting conditions for the
        reconstruction.

        Args
        ----

        Returns
        -------

        """

        if use_multiple_vertex_seeds:
            # Get additional vertex seeds using cylindrical geometry
            vertex_seeds = get_vertex_seeds(vertex_seed, track_dir_seed)

            if prescan_time:
                # For each vertex seed, we should optimize the track time.
                # i.e. we use the best-matching time for each vertex reconstruction
                seed_times = get_track_time_v(track_dir_seed, vertex_seeds, track_time, data)
                logls, dirs, verts = reconstruct_event_vt(vertex_seeds, track_dir_seed, seed_times, data)

            else:
                # Do not perform additional time matching. We reconstruct
                # each vertex seed with a fixed intial track time.
                logls, dirs, verts = reconstruct_event_v(vertex_seeds, track_dir_seed, track_time, data)
                seed_times = jnp.ones(vertex_seeds.shape[0]) * track_time

            # The solution is given by the fit with the best likelihood value
            # across all fits performed.
            ix = jnp.argmin(logls)
            return logls[ix], dirs[ix], verts[ix], seed_times[ix]

        # We are using only a single vertex seed
        if prescan_time:
            # Update time with best-match for given vertex_seed
            track_time = get_track_time(track_dir_seed, vertex_seed, track_time, data)

        logl, direction, vertex = reconstruct_event(vertex_seed, track_dir_seed, track_time, data)
        return logl, direction, vertex, track_time

    if use_batches:
        # The fitter function should operate on batches of events, i.e.
        # reconstruct multiple events in parallel.
        # All argument tensors and result tensor have one additional
        # batch dimension at index 0.
        run_reconstruction_v = jax.vmap(
                                    run_reconstruction,
                                    (0, 0, 0, 0),
                                    (0, 0, 0, 0)
                                )

        return run_reconstruction_v

    return run_reconstruction
