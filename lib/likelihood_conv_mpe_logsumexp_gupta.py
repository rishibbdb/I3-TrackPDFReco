from lib.gupta import c_multi_gupta_mpe_logprob_midpoint2_stable_v, c_multi_gupta_mpe_logprob_midpoint2_stable_fast_v
import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as norm_pdf
from jax.scipy.stats.norm import logpdf as norm_logpdf

def get_neg_c_triple_gamma_llh(eval_network_doms_and_track_fn, sigma=3.0):
    """
    here would be a smart docstring
    """

    sigma = jnp.array(sigma) # width of gaussian convolution

    @jax.jit
    def neg_c_triple_gamma_llh(track_direction,
                               track_vertex,
                               track_time,
                               event_data):


        # Constant parameters.
        sigma_noise = jnp.array(500.0)

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        #n_photons = jnp.round(charges + 0.5)
        #n_photons = jnp.clip(n_photons, min=1, max=1000)
        n_photons = jnp.clip(charges, min=1, max=10000)

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)
        

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.

        log_mix_probs = jax.nn.log_softmax(logits)
        log_physics_probs = c_multi_gupta_mpe_logprob_midpoint2_stable_v(delay_time,
                    log_mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma)


        log_floor_df = jnp.log(jnp.array(1./6000.))
        floor_weight = jnp.array(1.e-2)

        log_probs = jnp.concatenate([
                                        jnp.expand_dims(log_physics_probs, axis=0),
                                        jnp.expand_dims(jnp.ones_like(log_physics_probs) * log_floor_df, axis=0)
                                    ],
                                    axis=0
                                )

        weight = jnp.expand_dims(jnp.array([1.0-floor_weight, floor_weight]), axis=1)

        return -2.0 * jnp.sum(jax.scipy.special.logsumexp(log_probs, 0, weight))

    return neg_c_triple_gamma_llh



def get_llh_and_grad_fs_for_iminuit_migrad(eval_network_doms_and_track):
    """
    """
    neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

    @jax.jit
    def neg_llh_5D(x, track_time, data):
        track_direction = x[:2]
        track_vertex = x[2:]
        return neg_llh(track_direction, track_vertex, track_time, data)

    grad_neg_llh_5D = jax.jit(jax.grad(neg_llh_5D, argnums=0))

    return neg_llh_5D, grad_neg_llh_5D


def get_llh_and_grad_fs_for_iminuit_migrad_profile(eval_network_doms_and_track):
    """
    """
    neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

	# this gradient is 3D (vertex) not 5D (vertex + direction)
    grad_neg_llh_3D = jax.jit(jax.grad(neg_llh, argnums=1))

    return neg_llh, grad_neg_llh_3D


#Rishi
def get_neg_c_triple_gamma_llh_optimized(eval_network_doms_and_track_fn, sigma=3.0, precomputed_grid_data=None):
    """
    Optimized version that optionally uses precomputed grid.
    """
    sigma = jnp.array(sigma)
    
    @jax.jit
    def neg_c_triple_gamma_llh(track_direction, track_vertex, track_time, event_data):
        # Constant parameters.
        sigma_noise = jnp.array(500.0)
        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        
        n_photons = jnp.clip(charges, min=1, max=10000)
        
        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)
        
        log_mix_probs = jax.nn.log_softmax(logits)
        
        # Use optimized version
        log_physics_probs = c_multi_gupta_mpe_logprob_midpoint2_stable_fast_v(
            delay_time,
            log_mix_probs,
            av,
            bv,
            n_photons,
            sigma,
            precomputed_grid_data
        )
        
        log_floor_df = jnp.log(jnp.array(1./6000.))
        floor_weight = jnp.array(1.e-2)
        log_probs = jnp.concatenate([
            jnp.expand_dims(log_physics_probs, axis=0),
            jnp.expand_dims(jnp.ones_like(log_physics_probs) * log_floor_df, axis=0)
        ], axis=0)
        
        weight = jnp.expand_dims(jnp.array([1.0-floor_weight, floor_weight]), axis=1)
        return -2.0 * jnp.sum(jax.scipy.special.logsumexp(log_probs, 0, weight))
    
    return neg_c_triple_gamma_llh


from lib.gupta import build_weighted_logdf_srt_mpe
from lib.gupta import c_multi_gupta_mpe_logprob_midpoint2_stable_v
from lib.gupta import c_multi_gupta_spe_prob_large_sigma_fine_v
import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as norm_pdf
from jax.scipy.stats.norm import logpdf as norm_logpdf

def get_neg_c_triple_gamma_llh_SRT_noise(eval_network_doms_and_track_fn):
    
    @jax.jit
    def neg_c_triple_gamma_llh_SRT_noise(track_direction,
                                        track_vertex,
                                        track_time,
                                        event_data):
        
        # ---- hyperparams / constants
        sigma = jnp.array(3.0)  # width of gaussian convolution
        timeWindow = jnp.array(6000.0)  # ?
        floorWeight = jnp.array(1e-3)   # We use this when noiseModel is "none"
        sigma_noise = jnp.array(1000.0)

        # choose noiseModel: "none" | "flat" | "SRT"
        #noiseModel = "none"
        noiseModel = "SRT"
        modelStochastics = False

        # ---- unpack
        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        #n_photons = charges
        n_photons = jnp.clip(charges, a_min=1, a_max=10000)        
        '''
        print("###### event_data = ", event_data) 
        print("###### dom_pos.shape = ", dom_pos.shape)
        print("###### first_hit_times.shape = ", first_hit_times.shape)
        print("###### charges.shape = ", charges.shape)       
        print("###### n_photons.shape = ", n_photons.shape)
        '''
        # bare_pes/stoch_pes/expected_pes: suppose to be given
        # currentley dummy (charge from event data)
        ncol = event_data.shape[1]
        bare_pes = jnp.where(ncol > 5, event_data[:, 5], charges)
        stoch_pes = jnp.where(ncol > 6, event_data[:, 6], jnp.zeros_like(charges))
        expected_pes = jnp.where(ncol > 7, event_data[:, 7], bare_pes + stoch_pes)

        # ---- network
        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)
        '''
        print("###### logits.shape = ", logits.shape)
        print("###### av.shape = ", av.shape)
        print("###### bv.shape = ", bv.shape)
        print("###### geo_time.shape = ", geo_time.shape)
        '''
        # BuildWeightedPdf components 
        # barePdf = ConvolvedPdf(tres, ps, preJitter)
        # log_physics_probs
        log_mix_probs = jax.nn.log_softmax(logits)
        log_physics_probs = c_multi_gupta_mpe_logprob_midpoint2_stable_v(delay_time,
                    log_mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma)

        '''
        mix_probs = jnp.exp(log_mix_probs)
        log_noise_probs = jnp.log(c_multi_gupta_spe_prob_large_sigma_fine_v(delay_time,
                mix_probs,
                av,
                bv,
                sigma_noise))
        '''

        log_rDF = 0
        log_afterDF = 0
        
        log_floorDF = jnp.log(jnp.array(1./6000.))
        floorWeight = jnp.array(1.e-2) # to be optimized (so far 0.01)

        if (noiseModel != "none"):

            # rPdf = ConvolvedPdf(tres, random_noise_ps, 0)
            # need random_noise_ps
            # random_noise_ps = physics PDF smeared by ~1 μs
            # currently same (logits,av,bv) with log_physics_probs
            '''
            log_rDF = c_multi_gupta_spe_prob_large_sigma_fine_v( 
               delay_time, log_mix_probs, av, bv, n_photons, sigma_noise
            )
            '''
            log_physics_probs = c_multi_gupta_mpe_logprob_midpoint2_stable_v(delay_time,
                    log_mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma_noise)

            # floorPdf = 1/timeWindow
            log_floorDF = jnp.ones_like(delay_time) * (-jnp.log(timeWindow))

            # afterPdf = ConvolvedPdf(tres-2000, random_noise_ps, 0)
            '''
            log_afterDF = c_multi_gupta_spe_prob_large_sigma_fine_v(
                delay_time - 2000.0, log_mix_probs, av, bv, n_photons, sigma_noise
            )
            '''
            log_afterDF = c_multi_gupta_mpe_logprob_midpoint2_stable_v(delay_time - 2000.0,
                    log_mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma_noise)

        log_stochDF = 0
        if (modelStochastics):
            # stochPdf = ConvolvedPdf(tres, stochastics_ps, preJitter)
            # need stochastics_ps
            # currently same (logits,av,bv) with log_physics_probs
            log_stochDF = c_multi_gupta_spe_prob_large_sigma_fine_v(
                tres, log_mix_probs, av, bv, n_photons, preJitter
            )

        # final log df via SRT mixing
        log_df = build_weighted_logdf_srt_mpe(
            log_bareDF=log_physics_probs,
            log_stochDF=log_stochDF,
            log_rDF=log_rDF,
            log_afterDF=log_afterDF,
            log_floorDF=log_floorDF,
            bare_pes=bare_pes,
            stoch_pes=stoch_pes,
            expected_pes=expected_pes,
            modelStochastics=modelStochastics,
            noiseModel=noiseModel,
            floorWeight=floorWeight,
            low_pes=0.0,
        )

        # safety
        log_df = jnp.maximum(log_df, jnp.log(jnp.array(1e-300)))

        return -2.0 * jnp.sum(log_df)

    return neg_c_triple_gamma_llh_SRT_noise

