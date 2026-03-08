import os
from pathlib import Path
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_p
import pandas as pd

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import xwakes as xw

from cpymad.madx import Madx
from PyHEADTAIL.particles import Particles
import nafflib

class Analysis:
    def __init__(self, name):
        with h5py.File(f'data/{name}.h5', 'r') as f:
            self.n_macro_particles = f.attrs['n_macro_particles']
            self.n_turns = f.attrs['n_turns']
            self.nemitt_x_0 = f.attrs['nemitt_x_0']
            self.nemitt_y_0 = f.attrs['nemitt_y_0']
            
            self.gamma = f.attrs['gamma']
            self.beta = f.attrs['beta']
            
            self.qx = f.attrs['qx']
            self.qy = f.attrs['qy']
            self.dqx = f.attrs['dqx']
            self.dqy = f.attrs['dqy']
            
            self.x_tbt = f['x_tbt'][:]
            self.y_tbt = f['y_tbt'][:]
            self.px_tbt = f['px_tbt'][:]
            self.py_tbt = f['py_tbt'][:]

            self.sigma_x = f.attrs['sigma_x']
            self.sigma_y = f.attrs['sigma_y']
            self.mean_x = f.attrs['mean_x']
            self.mean_y = f.attrs['mean_y']

            self.twiss = f['twiss_data']
            
        df_twiss = pd.read_hdf(f'data/{name}.h5', key='twiss_data')

        self.betx = df_twiss['betx'].values
        self.bety = df_twiss['bety'].values
        self.alfx = df_twiss['alfx'].values
        self.alfy = df_twiss['alfy'].values

    def get_tune(self, xy):
        if xy == 'x':
            x_tbt = self.x_tbt
            p_tbt = self.px_tbt
            alpha = self.alfx[0]
            beta = self.betx[0]
        elif xy == 'y':
            x_tbt = self.y_tbt
            p_tbt = self.py_tbt
            alpha = self.alfy[0]
            beta = self.bety[0]
        else: # Do both
            Qx = get_tune('x')
            Qy = get_tune('y')
            return Qx, Qy
            
        x_norm = x_tbt / np.sqrt(beta)
        p_norm = (x_tbt * alpha / np.sqrt(beta) + p_tbt * np.sqrt(beta))
        z = x_norm - 1j * p_norm
        
        Q = []
        for i in range(self.n_macro_particles):
            signal = z[i, :] - np.mean(z[i, :])
            q = nafflib.get_tune(signal)
            Q.append(q)
        
        Q = np.array(Q)
        Q = np.where(Q < 0, 1 + Q, Q)

        if xy == 'x':
            self.Qx = Q
        else:
            self.Qy = Q
        return Q

    def tune_histogram(self, x_lims=None, y_lims=None, show=True):
        if not hasattr(self, 'Qx'):
            self.get_tune('x')
        if not hasattr(self, 'Qy'):
            self.get_tune('y')

        int_qx = np.floor(self.qx)
        int_qy = np.floor(self.qy)
        full_Qx = int_qx + self.Qx
        full_Qy = int_qy + self.Qy
       
        if x_lims is None:
            x_lims = (int_qx, int_qx + 1)
        else:
            x_lims = x_lims
        if y_lims is None:
            y_lims = (int_qy, int_qy + 1)
        else:
            y_lims = y_lims
        
        fig, ax = plt.subplots(figsize=(7, 7))
        h = ax.hist2d(full_Qx, full_Qy, bins=100, range=[x_lims, y_lims], cmap='Blues')
        #fig.colorbar(h[3], ax=ax, label='Count')
        
        ax.scatter(self.qx, self.qy, s=100, c='r', marker='*', zorder=5, label='Bare Tune')
        
        ax.set_xlabel(r"$Q_x$")
        ax.set_ylabel(r"$Q_y$")
        ax.set_title(f'Tune Footprint\nBare Tune: ({self.qx:.5f}, {self.qy:.5f})')
        
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        if show:
            plt.show()

        return fig, ax
