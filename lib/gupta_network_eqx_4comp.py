import numpy as np
import jaxlib
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import List, Tuple, Any
import os
import jax
import equinox as eqx

class TriplePandelNet(eqx.Module):
    layer0: eqx.Module
    layer1: eqx.Module
    layer2: eqx.Module
    layer3: eqx.Module
    layer4: eqx.Module
    layer5: eqx.Module
    layer6: eqx.Module

    def __init__(self, key, hidden_size=96):
        key = jax.random.split(key, 7)
        self.layer0 = eqx.nn.Linear(7, hidden_size, key=key[0])
        # self.layer1 = eqx.nn.Linear(hidden_size, hidden_size, key=key[1, :])
        # self.layer2 = eqx.nn.Linear(hidden_size, hidden_size, key=key[2, :])
        # self.layer3 = eqx.nn.Linear(hidden_size, hidden_size, key=key[3, :])
        # self.layer4 = eqx.nn.Linear(hidden_size, hidden_size, key=key[4, :])
        # self.layer5 = eqx.nn.Linear(hidden_size, hidden_size, key=key[5, :])
        # self.layer6 = eqx.nn.Linear(hidden_size+1, 12, key=key[6, :])
        self.layer1 = eqx.nn.Linear(hidden_size, hidden_size, key=key[1])
        self.layer2 = eqx.nn.Linear(hidden_size, hidden_size, key=key[2])
        self.layer3 = eqx.nn.Linear(hidden_size, hidden_size, key=key[3])
        self.layer4 = eqx.nn.Linear(hidden_size, hidden_size, key=key[4])
        self.layer5 = eqx.nn.Linear(hidden_size, hidden_size, key=key[5])
        self.layer6 = eqx.nn.Linear(hidden_size+1, 12, key=key[6])



    def eval(self, x):
        # x = [dist, cos(rho), sin(rho), z, cos(zenith),
        # sin(zenith)*cos(azimuth), sin(zenith)*sin(azimuth)]
        # dist, z in units of km
        d = x[0:1]

        x = jnp.tanh(self.layer0(x))
        x = jnp.tanh(self.layer1(x)) + x
        x = jnp.tanh(self.layer2(x)) + x
        x = jnp.tanh(self.layer3(x)) + x
        x = jnp.tanh(self.layer4(x)) + x
        x = jnp.tanh(self.layer5(x)) + x

        # outputs
        y = self.layer6(jnp.concatenate([x, d], axis=-1))

        # ordering used for training
        to_be_b = y[8:] + jnp.array([-7.62061505, -6.69140313, -5.70345812, -3.89233063])

        x = jnp.concatenate([y[:8], to_be_b], axis=-1)
        return x

    def __call__(self, x):
        return self.eval(x)

def get_network_eval_fn(bpath: str = '../../data/network/', n_layer=6, dtype=jnp.float32, n_hidden=96):
    """
    """

    key = jax.random.PRNGKey(0)
    model = TriplePandelNet(key, n_hidden)
    # the model is 32bit on disk.
    model = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
    model = eqx.tree_deserialise_leaves(bpath, model)
    # cast to desired precision.
    model = jtu.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, model)

    eval_network = model.__call__

    #params = []

    #for i in range(0, n_layer):
    #    layer_weights = np.load(os.path.join(bpath, f'dense_{i}_weights.npy'))
    #    layer_bias = np.load(os.path.join(bpath, f'dense_{i}_bias.npy'))
    #    params.append((
    #                    jnp.array(layer_weights, dtype=dtype),
    #                    jnp.array(layer_bias, dtype=dtype)
    #                ))

    #params = tuple(params)

    #def eval_network(x):
    #    """
    #    """

    #    x = jnp.tanh(jnp.dot(x, params[0][0]) + params[0][1])

    #    # residual block 1
    #    y = jnp.tanh(jnp.dot(x, params[1][0]) + params[1][1])
    #    x = jnp.tanh(jnp.dot(y, params[2][0]) + params[2][1]) + x

    #    # residual block 2
    #    y = jnp.tanh(jnp.dot(x, params[3][0]) + params[3][1])
    #    x = jnp.tanh(jnp.dot(y, params[4][0]) + params[4][1]) + x

    #    # outputs
    #    z = jnp.dot(x, params[5][0]) + params[5][1]

    #    return z

    return eval_network


def get_network_eval_v_fn(bpath: str = '../../data/network/', n_layer=6, dtype=jnp.float32, n_hidden=96):
    """
    """
    eval_network = get_network_eval_fn(bpath = bpath, n_layer=n_layer, dtype=dtype, n_hidden=n_hidden)
    eval_network_v = jax.jit(jax.vmap(eval_network, 0, 0))
    return eval_network_v

def get_network_eval_fn_f32(bpath: str = '../../data/network/', n_layer=6, dtype=jnp.float32, n_hidden=96):
    """
    """

    key = jax.random.PRNGKey(0)
    model = TriplePandelNet(key, n_hidden)
    # the model is 32bit on disk.
    # model = jtu.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
    model = eqx.tree_deserialise_leaves(bpath, model)
    # cast to desired precision.
    model = jtu.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, model)

    eval_network = model.__call__

    #params = []

    #for i in range(0, n_layer):
    #    layer_weights = np.load(os.path.join(bpath, f'dense_{i}_weights.npy'))
    #    layer_bias = np.load(os.path.join(bpath, f'dense_{i}_bias.npy'))
    #    params.append((
    #                    jnp.array(layer_weights, dtype=dtype),
    #                    jnp.array(layer_bias, dtype=dtype)
    #                ))

    #params = tuple(params)

    #def eval_network(x):
    #    """
    #    """

    #    x = jnp.tanh(jnp.dot(x, params[0][0]) + params[0][1])

    #    # residual block 1
    #    y = jnp.tanh(jnp.dot(x, params[1][0]) + params[1][1])
    #    x = jnp.tanh(jnp.dot(y, params[2][0]) + params[2][1]) + x

    #    # residual block 2
    #    y = jnp.tanh(jnp.dot(x, params[3][0]) + params[3][1])
    #    x = jnp.tanh(jnp.dot(y, params[4][0]) + params[4][1]) + x

    #    # outputs
    #    z = jnp.dot(x, params[5][0]) + params[5][1]

    #    return z

    return eval_network


def get_network_eval_v_fn_f32(bpath: str = '../../data/network/', n_layer=6, dtype=jnp.float32, n_hidden=96):
    """
    """
    eval_network = get_network_eval_fn_f32(bpath = bpath, n_layer=n_layer, dtype=dtype, n_hidden=n_hidden)
    eval_network_v = jax.jit(jax.vmap(eval_network, 0, 0))
    return eval_network_v