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
            
            self.x_scbsc = f['x_scbsc'][:]
            self.y_scbsc = f['y_scbsc'][:]
            self.px_scbsc = f['px_scbsc'][:]
            self.py_scbsc = f['py_scbsc'][:]

            self.sigma_x = f.attrs['sigma_x']
            self.sigma_y = f.attrs['sigma_y']
            self.mean_x = f.attrs['mean_x']
            self.mean_y = f.attrs['mean_y']

            self.n_interactions = f.attrs['n_interactions']
            self.beam_intensity = f.attrs['beam_intensity']

            self.twiss = f['twiss_data']
            
        df_twiss = pd.read_hdf(f'data/{name}.h5', key='twiss_data')

        self.betx = df_twiss['betx'].values
        self.bety = df_twiss['bety'].values
        self.alfx = df_twiss['alfx'].values
        self.alfy = df_twiss['alfy'].values

    def get_tune(self, xy):
        if xy == 'x':
            x_scbsc = self.x_scbsc
            p_scbsc = self.px_scbsc
            alpha = self.alfx[0]
            beta = self.betx[0]
        elif xy == 'y':
            x_scbsc = self.y_scbsc
            p_scbsc = self.py_scbsc
            alpha = self.alfy[0]
            beta = self.bety[0]
        else: # Do both
            Qx = get_tune('x')
            Qy = get_tune('y')
            return Qx, Qy
            
        x_norm = x_scbsc / np.sqrt(beta)
        p_norm = (x_scbsc * alpha / np.sqrt(beta) + p_scbsc * np.sqrt(beta))
        z = x_norm - 1j * p_norm
        
        Q = []
        for i in range(self.n_macro_particles):
            signal = z[i, :] - np.mean(z[i, :])
            q = nafflib.get_tune(signal) * self.n_interactions
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

        if x_lims is None:
            x_lims = (np.floor(self.qx), np.floor(self.qx) + 1)
        else:
            x_lims = x_lims
        if y_lims is None:
            y_lims = (np.floor(self.qy), np.floor(self.qy) + 1)
        else:
            y_lims = y_lims
        
        fig, ax = plt.subplots(figsize=(7, 7))
        h = ax.hist2d(self.Qx, self.Qy, bins=100, range=[x_lims, y_lims], cmap='Blues')
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
