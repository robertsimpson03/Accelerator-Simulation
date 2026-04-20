import os
from pathlib import Path
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_p, epsilon_0, c
import pandas as pd
from types import SimpleNamespace


import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import xwakes as xw

from cpymad.madx import Madx
from PyHEADTAIL.particles import Particles
import nafflib


def load_simulation(filename):
    namespace = SimpleNamespace()
    with h5py.File(filename, 'r') as f:
        for key in f.attrs.keys():
            setattr(namespace, key, f.attrs[key])
            
        for key in f.keys():
            if 'pandas_type' in f[key].attrs:
                val = pd.read_hdf(filename, key=key)
            else:
                val = f[key][()]
            setattr(namespace, key, val)
    return namespace

def get_tunes(x, y, px, py, twiss_df_at_mon, survivor_mask, half=0):
    # half is which half of the signl you wish to sample from
    # 1 is first half, 2 is second, 0 means use whole signal

    betx = twiss_df_at_mon['betx'].values
    bety = twiss_df_at_mon['bety'].values
    alfx = twiss_df_at_mon['alfx'].values
    alfy = twiss_df_at_mon['alfy'].values
    co_x = twiss_df_at_mon['x'].values
    co_y = twiss_df_at_mon['y'].values
    co_px = twiss_df_at_mon['px'].values
    co_py = twiss_df_at_mon['py'].values

    n_monitors = len(twiss_df_at_mon)
    n_turns = len(x.T)//n_monitors

    tunes=[]
    for q_unmaksed, p_unmasked, beta, alpha, co_q, co_p in zip([x, y], 
                                                               [px, py], 
                                                               [betx, bety], 
                                                               [alfx, alfy],
                                                               [co_x, co_y],
                                                               [co_px, co_py]):
        q = q_unmaksed
        p = p_unmasked

        midpoint = np.shape(q.T)[0]//2
        if half==1:
            q = (q.T[:midpoint]).T
            p = (p.T[:midpoint]).T
            n_turns = n_turns//2
        elif half==2:
            q = q[midpoint:]
            p = p[midpoint:]
            n_turns = n_turns//2

        beta_s = np.array(np.tile(beta, n_turns))
        alpha_s = np.array(np.tile(alpha, n_turns))
        co_q_s = np.array(np.tile(co_q, n_turns))
        co_p_s = np.array(np.tile(co_p, n_turns))

        q_norm =  (q - co_q_s) / np.sqrt(beta_s)
        p_norm = (q - co_q_s) * alpha_s / np.sqrt(beta_s) + (p - co_p_s) * np.sqrt(beta_s)
        z = q_norm - 1j * p_norm

        Q_total = []
        for i in range(len(x)):
            if survivor_mask[i]:
                signal = z[i, :] - np.mean(z[i, :])
                q_found = nafflib.get_tune(signal) * n_monitors
                Q_total.append(np.abs(q_found))
            else:
                Q_total.append(np.nan)
        tunes.append(np.array(Q_total))
    
    return tunes

