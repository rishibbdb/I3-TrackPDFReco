import jax
import jax.numpy as jnp
import optimistix as optx
from lib.experimental_methods import get_vertex_seeds

def get_scanner(neg_llh,
                use_multiple_vertex_seeds=False,
                prescan_time=False,
                scale=100.,
                rtol=1e-8,
                atol=1e-4,
                use_jit=True,
                n_splits=20):
    """Creates a scanner() function that performs a 2D profile likelihood
    scan in the sky (direction).
    Note: The argumenst are used globally within functions defined in the body.

    Arguments
    ---------
        neg_llh: A valid negative log likelihood function.
        use_multiple_vertex_seeds: If True, at each directional grid point,
            the scanner will perform the vertex optimization from
            additional vertex seeds.
            This increases robustness against local minima
            but increases the run-time
        prescan_time: If True, the scanner will search for the best time
            for a given vertex seed.
            This can improve convergence at the cost of run-time.
        scale: Re-scales the coordinate units during optimization.
            (maps step size of 1 to value of scale).
        rtol: relative tolerance of the optimizer (see optimistix.BFGS)
        atol: abolsute tolerance of the optimizer (see optimistix.BFGS)

    Returns
    -------
        scanner: A function that performs the likelihood scan
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

    # Define the likelihood function for the vertex optimization
    def neg_llh_vertex(x, args):
        """Defines the likelihood as function of the vertex.
        Consider any other arguments (args) as constant.

        Args
        ----
            x: jnp.array([x, y, z]), the current vertex.
            args: a tuple (track_dir, track_time, data)

        Returns
        -------
            Negative log-likelihood value.
        """
        pars, data = args
        track_dir = pars[:3]
        track_time = pars[3]

        x_prime = x * scale

        return neg_llh(track_dir, x_prime, track_time, data)

    def reconstruct_vertex(vertex_seed, track_dir, track_time, data):
        """Performs a single vertex reconstruction.

        Args
        ----
        vertex_seed: jnp.array([x, y, z]) in m
        track_dir: jnp.array([zenith, azimuth]) in radians
        track_time: jnp.array(t) in ns
        data: jnp.array(data) with shape
                (n_sensors, n_features) = (N, 5)

        Returns
        -------
            Best-fit Negative loglikelihood value (neglogl) and
            corresponding vertex as tuple(jnp.array, jnp.array([x,y,z]).
        """
        solver = optx.BestSoFarMinimiser(optx.BFGS(rtol=rtol, atol=atol, use_inverse=True))
        pars = jnp.concatenate([track_dir, jnp.expand_dims(track_time, axis=0)])
        args = (pars, data)
        sol = optx.minimise(neg_llh_vertex,
                            solver,
                            vertex_seed / scale,
                            args=args,
                            throw=False).value

        return neg_llh_vertex(sol, args), sol * scale

    # Vectorize over vertex argument
    reconstruct_vertex_v = jax.vmap(reconstruct_vertex, (0, None, None, None), 0)

    # Vectorize over vertex and time arguments
    reconstruct_vertex_vt = jax.vmap(reconstruct_vertex, (0, None, 0, None), 0)

    def run_reconstruction(track_dir, vertex_seed, track_time, data):
        """Wraps a single vertex reconstruction for a given track direction.
        Allows reconstructing that vertex multiple times with different
        seed values. And provides possibility to adjust the corresponding
        time constant to provide best starting conditions for the
        reconstruction.

        Args
        ----

        Returns
        -------

        """

        if use_multiple_vertex_seeds:
            # Get additional vertex seeds using cylindrical geometry
            vertex_seeds = get_vertex_seeds(vertex_seed, track_dir)

            if prescan_time:
                # For each vertex seed, we should optimize the track time.
                # i.e. we use the best-matching time for each vertex reconstruction
                seed_times = get_track_time_v(track_dir, vertex_seeds, track_time, data)
                logls, verts = reconstruct_vertex_vt(vertex_seeds, track_dir, seed_times, data)

            else:
                # Do not perform additional time matching. We reconstruct
                # each vertex seed with a fixed intial track time.
                logls, verts = reconstruct_vertex_v(vertex_seeds, track_dir, track_time, data)
                seed_times = jnp.ones(vertex_seeds.shape[0]) * track_time

            # The solution is given by the fit with the best likelihood value
            # across all fits performed.
            ix = jnp.argmin(logls)
            return logls[ix], verts[ix], seed_times[ix]

        # We are using only a single vertex seed
        if prescan_time:
            # Update time with best-match for given vertex_seed
            track_time = get_track_time(track_dir, vertex_seed, track_time, data)

        logl, vertex = reconstruct_vertex(vertex_seed, track_dir, track_time, data)
        return logl, vertex, track_time

    # Vectorize directions vertices in grid
    run_reconstruction_v = jax.vmap(run_reconstruction, (0, None, None, None), 0)
    # Placeholder. We may jit later if user requests it.
    run_reconstruction_v_jit = run_reconstruction_v

    def run_profile_llh_scan(grid_x,
                            grid_y,
                            vertex_seed,
                            track_time,
                            data):
        """Runs multiple vertex reconstructions (one per direction point
        within the grid specified by the function arguments). Depending on
        global parameters, each vertex reconstruction may be obtained
        by doing multiple optimizations starting from different seeds.

        Args
        ----

        Returns
        -------

        """

        # Flatten input grid, so that grid points are
        # can be indexed in 0-th position of the scan_dirs array.
        scan_dirs = jnp.column_stack([grid_x.flatten(), grid_y.flatten()])
        if n_splits < 2:
            # No splitting of grid. We process everything all at once.
            if use_jit:
                run_reconstruction_v_jit = \
                    jax.jit(run_reconstruction_v).lower(scan_dirs, vertex_seed, track_time, data).compile()

            logls, sol_pos, sol_time = \
                run_reconstruction_v_jit(scan_dirs, vertex_seed, track_time, data)

        else:
            # Process the grid in n_split batches.
            # This can be helpful if the grid does not fit
            # within GPU memory. (avoids OOM error)
            # Note: the number of gridpoints needs to be
            # divisible by n_splits.
            n_per_split, r = divmod(len(scan_dirs), n_splits)
            assert r==0, ("The number of grid points need to be divisible "
                          "by number of batches (n_splits).")

            # Reconstruct each batch and collect results.
            logls = []
            sol_pos = []
            sol_time = []
            for i in range(n_splits):
                current_scan_dirs = scan_dirs[i*n_per_split: (i+1) * n_per_split, :]
                if use_jit:
                    run_reconstruction_v_jit = \
                        jax.jit(run_reconstruction_v).lower(
                            current_scan_dirs,
                            vertex_seed,
                            track_time,
                            data
                        ).compile()

                logls_, sol_pos_, sol_time_ = \
                    run_reconstruction_v_jit(
                        current_scan_dirs,
                        vertex_seed,
                        track_time,
                        data
                    )

                logls.append(logls_)
                sol_pos.append(sol_pos_)
                sol_time.append(sol_time_)

            # Combine batches.
            logls = jnp.concatenate(logls, axis=0)
            sol_pos = jnp.concatenate(sol_pos, axis=0)
            sol_time = jnp.concatenate(sol_time, axis=0)

        # Restore shape of results to match
        # the one of the input grid.
        sol_time = sol_time.reshape(grid_x.shape)
        logls = logls.reshape(grid_x.shape)
        sol_x = sol_pos[:, 0].reshape(grid_x.shape)
        sol_y = sol_pos[:, 1].reshape(grid_x.shape)
        sol_z = sol_pos[:, 2].reshape(grid_x.shape)
        sol_vertex = jnp.concatenate(
            [
                jnp.expand_dims(sol_x, axis=-1),
                jnp.expand_dims(sol_y, axis=-1),
                jnp.expand_dims(sol_z, axis=-1),
            ],
            axis = -1
        )
        return logls, sol_vertex, sol_time

    # Users interact with the code via the run_profile_llh_scan function.
    return run_profile_llh_scan
