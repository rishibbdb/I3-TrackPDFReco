#!/usr/bin/env python

import pickle
from typing import List, Any
import numpy as np

def get_bin_idx(val: float, bins: np.ndarray) -> int:
    assert np.logical_and(val > bins[0], val < bins[-1]), f'value {val} not within bounds [{bins[0]}, {bins[-1]}]'
    return np.digitize(val, bins, right=False)-1

def adjust_plot_1d(fig, ax, plot_args=None):
    if not plot_args:
        plot_args = {}

    for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(1.5)
          ax.spines[axis].set_color('0.0')

    y_scale_in_log = plot_args.get('y_axis_in_log', False)
    if(y_scale_in_log):
        ax.set_yscale('log')

    ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=18)
    ax.yaxis.set_ticks_position('both')
    ax.set_ylabel(plot_args.get('ylabel', 'pdf'), fontsize=20)
    ax.set_xlabel(plot_args.get('xlabel', 'var 1'), fontsize=20)
    ax.set_ylim(plot_args.get('ylim', [0, 1]))
    ax.set_xlim(plot_args.get('xlim', [0, 1]))
    ax.legend()

def load_table_from_pickle(infile: str) -> List[Any]:
    table = pickle.load(open(infile, "rb"))
    bin_info = dict()
    bin_info['dist'] = {'c': table['bin_centers'][0],
            'e': table['bin_edges'][0],
            'w': table['bin_widths'][0]}

    bin_info['rho'] = {'c': table['bin_centers'][1],
            'e': table['bin_edges'][1],
            'w': table['bin_widths'][1]}

    bin_info['z'] = {'c': table['bin_centers'][2],
            'e': table['bin_edges'][2],
            'w': table['bin_widths'][2]}

    bin_info['dt'] = {'c': table['bin_centers'][3],
            'e': table['bin_edges'][3],
            'w': table['bin_widths'][3]}

    return table, bin_info
