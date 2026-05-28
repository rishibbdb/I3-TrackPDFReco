import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp

"""
Fast approximation to lower incomplete gamma function.
Implementation follows the formulas given in

Geosci. Model Dev., 3, 329–336, 2010
www.geosci-model-dev.net/3/329/2010/
doi:10.5194/gmd-3-329-2010

by U. Blahak.
"""

# from Table 1.
p_coeffs = jnp.array([
    9.4368392235e-3,
    -1.0782666481e-4,
    -5.8969657295e-6,
    2.8939523781e-7,
    1.0043326298e-1,
    5.5637848465e-1
])

q_coeffs = jnp.array([
    1.1464706419e-1,
    2.6963429121,
    -2.9647038257,
    2.1080724954
])

r_coeffs = jnp.array([
    0.0,
    1.1428716184,
    -6.6981186438e-3,
    1.0480765092e-4
])

s_coeffs = jnp.array([
    1.0480765092,
    2.3423452308,
    -3.6174503174e-1,
    -3.1376557650,
    2.9092306039
])


def c_coeffs(a):
    """
    eq. 14-17
    """
    ap2 = jnp.power(a, 2)
    ap3 = jnp.power(a, 3)
    ap4 = jnp.power(a, 4)

    c1 = 1.0 + p_coeffs[0]*a + p_coeffs[1]*ap2 + p_coeffs[2]*ap3 + p_coeffs[3]*ap4 + p_coeffs[4]*(jnp.exp(-p_coeffs[5]*a)-1.0)
    c2 = q_coeffs[0] + q_coeffs[1]/a + q_coeffs[2]/ap2 + q_coeffs[3]/ap3
    c3 = r_coeffs[0] + r_coeffs[1]*a + r_coeffs[2]*ap2 + r_coeffs[3]*ap3
    c4 = s_coeffs[0] + s_coeffs[1]/a + s_coeffs[2]/ap2 + s_coeffs[3]/ap3 + s_coeffs[4]/ap4
    return jnp.array([c1, c2, c3, c4])


def tanh_approx(x):
    """
    eq. 23.
    an approximation to tanh.
    does not appear faster than the direct call.
    """
    ct = 9.37532
    crit = ct / 3
    y = (9*ct**2*x + 27*x**3) / (ct**3 + 27*ct*x**2)
    y = jnp.where(x <= -crit, -1.0, y)
    y = jnp.where(x >= crit, 1.0, y)
    return y


def regularized_lower_incomplete_gamma_approx(x, a):
    """
    note: the original paper treats the lower incomplete gamma function.
    The regularized lower incomplete gamma function differs by the
    normalization factor 1/Gamma[a].
    """

    c = c_coeffs(a)

    # eq. 13
    w = 0.5 + 0.5 * jnp.tanh(c[1]*(x-c[2]))

    # alternatively, use approximate tanh.
    #w = 0.5 + 0.5 * tanh_approx(c[1]*(x-c[2]))

    # eq. 12
    r1 = 1.0/jax.scipy.special.gamma(a) * jnp.exp(-x)*jnp.power(x, a)*(1.0/a + c[0]*x/(a*(a+1)) + (c[0]*x)**2/(a*(a+1)*(a+2)))*(1-w)
    r2 = w*(1.0-jnp.power(c[3], -x))
    return r1+r2


def regularized_lower_incomplete_gamma_approx_w_existing_coefficients(x, a, c):
    """
    note: the original paper treats the lower incomplete gamma function.
    The regularized lower incomplete gamma function differs by the
    normalization factor 1/Gamma[a].
    """

    # eq. 13
    w = 0.5 + 0.5 * jnp.tanh(c[1]*(x-c[2]))

    # alternatively, use approximate tanh.
    #w = 0.5 + 0.5 * tanh_approx(c[1]*(x-c[2]))

    # eq. 12
    r1 = 1.0/jax.scipy.special.gamma(a) * jnp.exp(-x)*jnp.power(x, a)*(1.0/a + c[0]*x/(a*(a+1)) + (c[0]*x)**2/(a*(a+1)*(a+2)))*(1-w)
    r2 = w*(1.0-jnp.power(c[3], -x))
    return r1+r2

def gamma_cdf_fast(x, a, b):
    return jnp.clip(regularized_lower_incomplete_gamma_approx(x*b, a), min=0.0, max=1.0)

def gamma_sf_fast(x, a, b):
    return jnp.clip(1.0-regularized_lower_incomplete_gamma_approx(x*b, a), min=0.0, max=1.0)

def gamma_sf_fast_w_existing_coefficients(x, a, b, c):
    return jnp.clip(1.0-regularized_lower_incomplete_gamma_approx_w_existing_coefficients(x*b, a, c), min=0.0, max=1.0)

#def log1mexp(x):
#    return jnp.where(
#            x < jnp.log(0.5), jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x))
#    )

def log1mexp(x):
    """
    Numerically stable calculation
    of the quantity log(1 - exp(x)),
    following the algorithm of
    Machler [1]. This is
    the algorithm used in TensorFlow Probability,
    PyMC, and Stan, but it is not provided
    yet with Numpyro.

    Currently returns NaN for x > 0,
    but may be modified in the future
    to throw a ValueError

    [1] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    # return 0. rather than -0. if
    # we get a negative exponent that exceeds
    # the floating point representation
    crit = -0.6931472
    arr_x = 1.0 * jnp.array(x)

    crit_oob = jnp.log(jnp.finfo(
        arr_x.dtype).smallest_normal)+5
    oob = arr_x < crit_oob
    mask = arr_x > crit

    more_val = jnp.log(-jnp.expm1(jnp.clip(arr_x, min=crit)))
    less_val = jnp.log1p(-jnp.exp(jnp.clip(arr_x, max=crit)))

    return jnp.where(
        oob,
        -jnp.exp(crit_oob),
        jnp.where(
            mask,
            more_val,
            less_val))

def logW(x, c):
    x = jnp.clip(c[1] * (x-c[2]), min=-15, max=15)
    _x = jnp.concatenate([jnp.expand_dims(x, axis=0), jnp.expand_dims(-x, axis=0)],
                         axis=0)
    return x - logsumexp(_x, 0)

def log_regularized_lower_incomplete_gamma_approx(x, a):
    c = c_coeffs(a)
    lw = logW(x, c) # log W
    l1mw = log1mexp(lw)  # log (1-W)
    log_r1 = l1mw - x + a*jnp.log(x) - jax.scipy.special.gammaln(a) + jnp.log((1.0/a +
                                                                        c[0]*x/(a*(a+1)) +
                                                                        (c[0]*x)**2/(a*(a+1)*(a+2))))
    log_r2 = lw + log1mexp(-x * jnp.log(c[3]))

    x = jnp.concatenate([jnp.expand_dims(log_r1, axis=0), jnp.expand_dims(log_r2, axis=0)],
                        axis=0)
    return logsumexp(x, 0)

def log_gamma_cdf_fast(x, a, b):
    return log_regularized_lower_incomplete_gamma_approx(x*b, a)

def log_gamma_sf_fast(x, a, b):
    return log1mexp(log_gamma_cdf_fast(x, a, b))


