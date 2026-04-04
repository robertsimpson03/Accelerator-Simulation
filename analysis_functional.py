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
    betx = twiss_df_at_mon['betx'].values
    bety = twiss_df_at_mon['bety'].values
    alfx = twiss_df_at_mon['alfx'].values
    alfy = twiss_df_at_mon['alfy'].values

    n_monitors = len(twiss_df_at_mon)
    n_turns = len(x.T)//n_monitors

    tunes=[]
    for q_unmaksed, p_unmasked, beta, alpha in zip([x, y], 
                                                   [px, py], 
                                                   [betx, bety], 
                                                   [alfx, alfy]):
        q = q_unmaksed[survivor_mask]
        p = p_unmasked[survivor_mask]

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

        q_norm =  q / np.sqrt(beta_s)
        p_norm = q * alpha_s / np.sqrt(beta_s) + p * np.sqrt(beta_s)
        z = q_norm - 1j * p_norm

        Q_total = []
        for i in range(np.sum(survivor_mask)):
            signal = z[i, :] - np.mean(z[i, :])
            q_found = nafflib.get_tune(signal) * n_monitors
            Q_total.append(np.abs(q_found))
        tunes.append(np.array(Q_total))
    
    return tunes

