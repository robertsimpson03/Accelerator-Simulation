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
            if self.n_macro_particles=='co':
                self.n_macro_particles = 1
            self.n_turns = f.attrs['n_turns']
            self.nemitt_x_0 = f.attrs['nemitt_x_0']
            self.nemitt_y_0 = f.attrs['nemitt_y_0']
            
            self.gamma = f.attrs['gamma']
            self.beta = f.attrs['beta']
            
            self.qx = f.attrs['qx']
            self.qy = f.attrs['qy']
            self.dqx = f.attrs['dqx']
            self.dqy = f.attrs['dqy']
            
            self.x = f['x'][:]
            self.y = f['y'][:]
            self.px = f['px'][:]
            self.py = f['py'][:]

            try:
                self.sigma_x = f['sigma_x'][:]
                self.sigma_y = f['sigma_y'][:]
                self.mean_x = f['mean_x'][:]
                self.mean_y = f['mean_y'][:]

                self.n_interactions = f.attrs['n_interactions']
                self.beam_intensity = f.attrs['beam_intensity']
            except:
                pass

            self.n_monitors = f.attrs['n_monitors']
            self.monitor_names = f.attrs['monitor_names']

            self.twiss = f['twiss_data']
            
        self.df_twiss = pd.read_hdf(f'data/{name}.h5', key='twiss_data')

        self.betx = self.df_twiss['betx'].values
        self.bety = self.df_twiss['bety'].values
        self.alfx = self.df_twiss['alfx'].values
        self.alfy = self.df_twiss['alfy'].values

    def get_tune(self, xy):
        mon_mask = self.df_twiss['name'].str.contains('mon')
        df_mon = self.df_twiss[mon_mask]

        if xy == 'x':
            x = np.array(self.x).T
            p = np.array(self.px).T
            beta = df_mon['betx'].values
            alpha = df_mon['alfx'].values
        else:
            x = np.array(self.y).T
            p = np.array(self.py).T
            beta = df_mon['bety'].values
            alpha = df_mon['alfy'].values

        beta_s = np.array(np.tile(beta, self.n_turns))
        alpha_s = np.array(np.tile(alpha, self.n_turns))

        x_norm =  x / np.sqrt(beta_s)
        p_norm = x * alpha_s / np.sqrt(beta_s) + p * np.sqrt(beta_s)
        z = x_norm - 1j * p_norm
        
        Q_total = []
        for i in range(self.n_macro_particles):
            signal = z[i, :] - np.mean(z[i, :])
            q_found = nafflib.get_tune(signal) * self.n_monitors
            Q_total.append(np.abs(q_found))
        result = np.array(Q_total)

        if xy == 'x':
            self.Qx = result
        else:
            self.Qy = result
            
        return result


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
        h = ax.hist2d(self.Qx, self.Qy, 
                      bins=100, range=[x_lims, y_lims], cmap='Blues')
        #fig.colorbar(h[3], ax=ax, label='Count')
        
        ax.scatter(self.qx, self.qy, 
                   s=100, c='r', marker='*', zorder=5, label='Bare Tune')
        
        ax.set_xlabel(r"$Q_x$")
        ax.set_ylabel(r"$Q_y$")
        ax.set_title(f'Tune Footprint\nBare Tune: ({self.qx:.5f}, {self.qy:.5f})')
        
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        if show:
            plt.show()


    """def plot_spectrum(self, field, i=0):
        if self.n_macro_particles==(1 or 'co'):
            if field=='x':
                field=self.x
            elif field=='y':
                field=self.y
            elif field=='px':
                field=self.px
            elif field=='py':
                field=self.py
        
        else:
            if field=='x':
                field=self.x[i]
            elif field=='y':
                field=self.y[i]
            elif field=='px':
                field=self.px[i]
            elif field=='py':
                field=self.py[i]
        
        N = len(field)
        dt = 1 / self.n_monitors
        fft = np.fft.fft(field) 
        freqs = np.fft.fftfreq(N, d=dt)

        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(freqs, abs(fft))
        ax.set_xlim(0, self.n_monitors/2)
        ax.set_xticks(np.arange(0, (self.n_monitors+1)/2, 2))"""

    def plot_spectrum(self, xy, i=0, n_peaks=5):
        import nafflib

        # 1. Get the normalized signal (same as your tune logic)
        if xy == 'x':
            coord, mom = np.array(self.x).T, np.array(self.px).T
            beta = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['betx'].values
            alpha = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['alfx'].values
        else:
            coord, mom = np.array(self.y).T, np.array(self.py).T
            beta = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['bety'].values
            alpha = self.df_twiss[self.df_twiss['name'].str.contains('mon')]['alfy'].values

        # Normalize
        beta_s = np.tile(beta, self.n_turns)
        alpha_s = np.tile(alpha, self.n_turns)
        z = (coord / np.sqrt(beta_s)) - 1j * (coord * alpha_s / np.sqrt(beta_s) + mom * np.sqrt(beta_s))
        
        signal = z[i, :] - np.mean(z[i, :])

        # 2. Use NAFF to find multiple peaks
        # nafflib.get_tunes returns (frequencies, amplitudes)
        # We multiply frequencies by n_monitors to get the Total Tune
        freqs, amplitudes, stuff = nafflib.get_tunes(signal, n_peaks)
        tunes = np.abs(freqs * self.n_monitors)

        # 3. Create a "Synthetic Spectrum" plot
        # Since NAFF is a frequency finder, we plot vertical lines (stems) 
        # representing the strength of each frequency it found.
        fig, ax = plt.subplots(figsize=(12, 5))
        
        markerline, stemlines, baseline = ax.stem(tunes, np.abs(amplitudes), 
                                                 linefmt='C0-', markerfmt='C0o', 
                                                 basefmt=" ")
        
        # Label the main peak
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
            coord, mom = np.array(self.x).T, np.array(self.px).T
            beta, alpha = df_mon['betx'].values, df_mon['alfx'].values
        else:
            coord, mom = np.array(self.y).T, np.array(self.py).T
            beta, alpha = df_mon['bety'].values, df_mon['alfy'].values

        beta_s = np.tile(beta, self.n_turns)
        alpha_s = np.tile(alpha, self.n_turns)
        
        z = (coord / np.sqrt(beta_s)) - 1j * (coord * alpha_s / np.sqrt(beta_s) + mom * np.sqrt(beta_s))

        N = z.shape[1]
        dt = 1.0 / len(beta) # Time step is 1/N_monitors
        freqs = np.fft.fftfreq(N, d=dt)
        
        total_magnitude = np.zeros(N)

        for i in range(self.n_macro_particles):
            signal = z[i, :] - np.mean(z[i, :])
            fft_val = np.fft.fft(signal)
            total_magnitude += np.abs(fft_val)

        avg_magnitude = total_magnitude / self.n_macro_particles

        fig, ax = plt.subplots(figsize=(20, 6))
        
        mask = (freqs > 0) & (freqs < 20) # Show up to Q=20 to see the first alias
        ax.plot(freqs[mask], avg_magnitude[mask], color='black', lw=1.2)
        
        #ax.set_yscale('log') # Spectra are usually best viewed in log scale
        plt.xticks(np.arange(0,20,1))
        if xy=='y':
            ax.axvline(13.0, color='red', linestyle='--', lw=0.2, label='13th Harmonic Resonance')
        ax.set_xlabel('Total Tune (Q)')
        ax.set_ylabel('Average Magnitude (Log Scale)')
        ax.set_title(f'Ensemble Average {xy} Spectrum (N={self.n_macro_particles})')
        ax.grid(True, which='both', alpha=0.3)
        
        plt.show()

    def plot_sextupole_spectrum(self, plane='y'):
        if plane == 'y':
            data = np.array(self.y).T # Shape: (n_particles, n_samples)
        else:
            data = np.array(self.x).T

        sextupole_moment = np.mean(data**3, axis=0)
        
        N = len(sextupole_moment)
        freqs = np.fft.rfftfreq(N, d=1/self.n_monitors)
        magnitudes = np.abs(np.fft.rfft(sextupole_moment - np.mean(sextupole_moment)))

        fig, ax = plt.subplots(figsize=(20, 6))
        
        ax.plot(freqs, magnitudes, color='black', lw=0.8)
        ax.axvline(13.0, color='red', linestyle='--', lw=0.6, alpha=0.7, label='13th Harmonic')

        ax.set_xlim(0.0, 20.0) # Focus on the 13th harmonic region
        
        ax.grid(which='major', color='#CCCCCC', lw=0.5)
        ax.grid(which='minor', color='#EEEEEE', lw=0.3, ls=':')
        
        plt.xticks(np.arange(0,20,1))
        ax.set_xlabel('Harmonic / Tune')
        ax.set_ylabel(r'Sextupole Moment Amplitude $\langle y^3 \rangle$')
        ax.set_title(f'Coherent Sextupole Spectrum (ISIS {plane}-plane)')
        ax.legend(frameon=False)
        
        plt.tight_layout()
        plt.show()
