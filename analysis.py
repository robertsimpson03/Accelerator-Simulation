import os
from pathlib import Path
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_p, epsilon_0, c
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
        with h5py.File(f'{name}.h5', 'r') as f:
            self.time = f.attrs['time']
            self.runtime = f.attrs['runtime']
            self.n_particles = f.attrs['n_particles']
            self.n_turns = f.attrs['n_turns']
            self.n_monitors = f.attrs['n_monitors']
            self.nemitt_x_0 = f.attrs['nemitt_x_0']
            self.nemitt_y_0 = f.attrs['nemitt_y_0']
            
            self.gamma = f.attrs['gamma']
            self.beta = f.attrs['beta']
            self.p0c = f.attrs['p0c'] 
            self.length = f.attrs['length']

            self.qx = f.attrs['qx']
            self.qy = f.attrs['qy']
            self.dqx = f.attrs['dqx']
            self.dqy = f.attrs['dqy']
            
            self.x = f['x'][:]
            self.y = f['y'][:]
            self.px = f['px'][:]
            self.py = f['py'][:]

            try:
                self.n_interactions = f.attrs['n_interactions']
                self.beam_intensity = f.attrs['beam_intensity']
                self.x0 = f['x0'][:]
                self.y0 = f['y0'][:]
                self.sigma_x = f['sigma_x'][:]
                self.sigma_y = f['sigma_y'][:]
                self.x_pipe = f['x_pipe'][:]
                self.y_pipe = f['y_pipe'][:]
                self.x_length = f['x_length'][:]
                self.y_length = f['y_length'][:]
            except:
                pass

            self.twiss = f['twiss_data']
            
        self.df_twiss = pd.read_hdf(f'{name}.h5', key='twiss_data')

        self.betx = self.df_twiss['betx'].values
        self.bety = self.df_twiss['bety'].values
        self.alfx = self.df_twiss['alfx'].values
        self.alfy = self.df_twiss['alfy'].values

    def get_tune(self, xy, centroid=False):
        mon_mask = self.df_twiss['name'].str.contains('mon')
        df_mon = self.df_twiss[mon_mask]

        if xy == 'x':
            x = self.x
            p = self.px
            beta = df_mon['betx'].values
            alpha = df_mon['alfx'].values
        else:
            x = self.y
            p = self.py
            beta = df_mon['bety'].values
            alpha = df_mon['alfy'].values

        if centroid==True:
            x = np.mean(x, axis=0)
            p = np.mean(p, axis=0)

        beta_s = np.array(np.tile(beta, self.n_turns))
        alpha_s = np.array(np.tile(alpha, self.n_turns))

        x_norm =  x / np.sqrt(beta_s)
        p_norm = x * alpha_s / np.sqrt(beta_s) + p * np.sqrt(beta_s)
        z = x_norm - 1j * p_norm

        if centroid==False:
            Q_total = []
            for i in range(self.n_particles):
                signal = z[i, :] - np.mean(z[i, :])
                q_found = nafflib.get_tune(signal) * self.n_monitors
                Q_total.append(np.abs(q_found))
            result = np.array(Q_total)
            if xy == 'x':
                self.Qx = result
            else:
                self.Qy = result
        else:
            signal = z - np.mean(z)
            result = np.abs(nafflib.get_tune(signal) * self.n_monitors)

            if xy == 'x':
                self.Qx_centroid = result
            else:
                self.Qy_centroid = result
            
        return result

    def max_tune_shift(self):
        k_e = 1/(4*np.pi*epsilon_0)
        _lambda = self.beam_intensity / self.length
        K_sc = (2*k_e*e*_lambda
                /(self.beta*self.gamma**2*self.p0c))

        sig_x = np.mean(self.sigma_x)
        sig_y = np.mean(self.sigma_y)
        R = self.length / (2*np.pi)

        DQx = K_sc*R**2/(2*sig_x*(sig_x+sig_y)*self.qx)
        DQy = K_sc*R**2/(2*sig_y*(sig_x+sig_y)*self.qy)

        return DQx, DQy


    def tune_histogram(self, x_lims=None, y_lims=None, centroid=False, show=True):
        if not hasattr(self, 'Qx'):
            self.get_tune('x')
        if not hasattr(self, 'Qy'):
            self.get_tune('y')
        if not hasattr(self, 'Qx_centroid'):
            self.get_tune('x', centroid=True)
        if not hasattr(self, 'Qy_centroid'):
            self.get_tune('y', centroid=True)

        if x_lims is None:
            x_lims = (np.floor(self.qx), np.floor(self.qx) + 1)
        else:
            x_lims = x_lims
        if y_lims is None:
            y_lims = (np.floor(self.qy), np.floor(self.qy) + 1)
        else:
            y_lims = y_lims
        
        fig, ax = plt.subplots(figsize=(7, 7))
        h = ax.hist2d(self.Qx, self.Qy, 
                      bins=100, range=[x_lims, y_lims], cmap='Blues')
        #fig.colorbar(h[3], ax=ax, label='Count')
        
        ax.scatter(self.qx, self.qy, 
                   s=100, c='r', marker='*', zorder=5, label='Bare Tune')
        if centroid==True:
            ax.scatter(self.Qx_centroid, self.Qy_centroid, 
                       s=100, c='b', marker='*', zorder=5, label='Centroid Tune')
        if hasattr(self, 'sigma_x'):
            self.DQx, self.DQy = self.max_tune_shift()
            ax.scatter(self.qx - self.DQx, self.qy - self.DQy, 
                       s=100, c='k', marker='*', zorder=5, label='Theoretical maximum tune shift')

        ax.set_xlabel(r"$Q_x$")
        ax.set_ylabel(r"$Q_y$")
        ax.set_title(f'Tune Footprint\nBare Tune: ({self.qx:.5f}, {self.qy:.5f})')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        if show:
            plt.show()


    def plot_spectrum(self, xy, i=0, n_peaks=5):
        import nafflib

        if xy == 'x':
            coord, mom = self.x, self.px
            beta = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['betx'].values
            alpha = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['alfx'].values
        else:
            coord, mom = self.y, self.py
            beta = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['bety'].values
            alpha = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['alfy'].values

        beta_s = np.tile(beta, self.n_turns)
        alpha_s = np.tile(alpha, self.n_turns)
        z = (coord / np.sqrt(beta_s)) - 1j * (coord * alpha_s / np.sqrt(beta_s) + mom * np.sqrt(beta_s))
        
        signal = z[i, :] - np.mean(z[i, :])

        freqs, amplitudes, stuff = nafflib.get_tunes(signal, n_peaks)
        tunes = np.abs(freqs * self.n_monitors)

        fig, ax = plt.subplots(figsize=(12, 5))
        
        markerline, stemlines, baseline = ax.stem(tunes, np.abs(amplitudes), 
                                                 linefmt='C0-', markerfmt='C0o', 
                                                 basefmt=" ")
        
        main_tune = tunes[0]
        ax.annotate(f'Main Tune: {main_tune:.4f}', 
                    xy=(main_tune, np.abs(amplitudes[0])), 
                    xytext=(main_tune+0.1, np.abs(amplitudes[0])),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        ax.set_xlabel('Total Tune (Q)')
        ax.set_ylabel('Amplitude (A.U.)')
        ax.set_title(f'NAFF Spectrum for {xy} (Top {n_peaks} Frequencies)')
        #ax.set_xlim(main_tune - 1, main_tune + 1) # Zoom in on the action
        ax.grid(True, alpha=0.3)
        print(tunes)
        plt.show()

    def plot_ensemble_fft_spectrum(self, xy):
        mon_mask = self.df_twiss['name'].str.contains('mon')
        df_mon = self.df_twiss[mon_mask]
        
        if xy == 'x':
            coord, mom = self.x, self.px
            beta, alpha = df_mon['betx'].values, df_mon['alfx'].values
        else:
            coord, mom = self.y, self.py
            beta, alpha = df_mon['bety'].values, df_mon['alfy'].values

        beta_s = np.tile(beta, self.n_turns)
        alpha_s = np.tile(alpha, self.n_turns)
        
        z = (coord / np.sqrt(beta_s)) - 1j * (coord * alpha_s / np.sqrt(beta_s) + mom * np.sqrt(beta_s))

        N = z.shape[1]
        dt = 1.0 / len(beta) 
        freqs = np.fft.fftfreq(N, d=dt)
        
        total_magnitude = np.zeros(N)

        for i in range(self.n_particles):
            signal = z[i, :] - np.mean(z[i, :])
            fft_val = np.fft.fft(signal)
            total_magnitude += np.abs(fft_val)

        avg_magnitude = total_magnitude / self.n_particles

        fig, ax = plt.subplots(figsize=(20, 6))
        
        mask = (freqs > 0) & (freqs < 20) 
        ax.plot(freqs[mask], avg_magnitude[mask], color='black', lw=1.2)
        
        plt.xticks(np.arange(0,20,1))
        if xy=='y':
            ax.axvline(13.0, color='red', linestyle='--', lw=0.2, label='13th Harmonic Resonance')
        ax.set_xlabel('Total Tune (Q)')
        ax.set_ylabel('Average Magnitude (Log Scale)')
        ax.set_title(f'Ensemble Average {xy} Spectrum (N={self.n_particles})')
        ax.grid(True, which='both', alpha=0.3)
        
        plt.show()

    def plot_sextupole_spectrum(self, plane='y'):
        if plane == 'y':
            data = np.array(self.y).T 
        else:
            data = np.array(self.x).T

        sextupole_moment = np.mean(data**3, axis=0)
        
        N = len(sextupole_moment)
        freqs = np.fft.rfftfreq(N, d=1/self.n_monitors)
        magnitudes = np.abs(np.fft.rfft(sextupole_moment - np.mean(sextupole_moment)))

        fig, ax = plt.subplots(figsize=(20, 6))
        
        ax.plot(freqs, magnitudes, color='black', lw=0.8)
        ax.axvline(13.0, color='red', linestyle='--', lw=0.6, alpha=0.7, label='13th Harmonic')

        ax.set_xlim(0.0, 20.0)
        
        ax.grid(which='major', color='#CCCCCC', lw=0.5)
        ax.grid(which='minor', color='#EEEEEE', lw=0.3, ls=':')
        
        plt.xticks(np.arange(0,20,1))
        ax.set_xlabel('Harmonic / Tune')
        ax.set_ylabel(r'Sextupole Moment Amplitude $\langle y^3 \rangle$')
        ax.set_title(f'Coherent Sextupole Spectrum (ISIS {plane}-plane)')
        ax.legend(frameon=False)
        
        plt.tight_layout()
        plt.show()

    def plot_field_in_pipe(self):
        fig, axs=plt.subplots(2, figsize=(20,12))
        axs[0].plot(np.zeros_like(self.x_pipe), c='k', ls='--')
        axs[0].plot(self.x_pipe + self.x_length/2, c='k')
        axs[0].plot(self.x_pipe - self.x_length/2, c='k')
        axs[0].plot(self.x0, c='r')
        axs[0].plot(self.x0 + self.sigma_x, c='r', ls='--')
        axs[0].plot(self.x0 - self.sigma_x, c='r', ls='--')

        axs[1].plot(np.zeros_like(self.x_pipe), c='k', ls='--')
        axs[1].plot(self.y_pipe + self.y_length/2, c='k')
        axs[1].plot(self.y_pipe - self.y_length/2, c='k')
        axs[1].plot(self.y0, c='r')
        axs[1].plot(self.y0 + self.sigma_y, c='r', ls='--')
        axs[1].plot(self.y0 - self.sigma_y, c='r', ls='--')

        axs[0].set_ylabel('1 sigma x beam in pipe')
        axs[1].set_ylabel('1 sigma y beam in pipe')
        plt.show()
