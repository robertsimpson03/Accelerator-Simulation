import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_p
import pandas as pd

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import xwakes as xw

from PyHEADTAIL.particles import Particles
import nafflib

def get_tune(x_tbt, p_tbt, beta, alpha):
    n_part = x_tbt.shape[0]
    n_turns = x_tbt.shape[1]

    x_norm = x_tbt / np.sqrt(beta)
    p_norm = (x_tbt * alpha / np.sqrt(beta) + p_tbt * np.sqrt(beta))
    
    z = x_norm - 1j * p_norm
    
    tunes = []
    
    for i in range(n_part):
        signal = z[i, :] - np.mean(z[i, :])
        q = nafflib.get_tune(signal)
        tunes.append(q)
            
    return np.array(tunes)


def calculate_emittance_tbt(x_tbt, px_tbt, beta, gamma):
    x_centered = x_tbt - np.mean(x_tbt, axis=0)
    px_centered = px_tbt - np.mean(px_tbt, axis=0)
    
    x2_avg = np.mean(x_centered**2, axis=0)
    px2_avg = np.mean(px_centered**2, axis=0)
    xpx_avg = np.mean(x_centered * px_centered, axis=0)
    
    geom_emitt = np.sqrt(x2_avg * px2_avg - xpx_avg**2)
    
    norm_emitt = beta * gamma * geom_emitt
    
    return norm_emitt


def plot_tunes(Qx, Qy, raw_Qx, raw_Qy, xl=0.00002, yl=0.00002, show=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    x_diff = Qx - raw_Qx % 1
    y_diff = Qy - raw_Qy % 1
    
    x_lims = (-xl/2, xl/2)
    y_lims = (-yl/2, yl/2)
    
    h = ax.hist2d(x_diff, y_diff, bins=50, range=[x_lims, y_lims], cmap='Blues')
    
    fig.colorbar(h[3], ax=ax, label='Count')
    
    ax.scatter(0, 0, s=100, c='r', marker='*', zorder=5)
    
    ax.set_xlabel(r"""$Q_x - Q_x^{\text{raw}}$
                  """ + fr"$Q_x^{{\text{{raw}}}}={raw_Qx%1}$")
    ax.set_ylabel(r"""$Q_y - Q_y^{\text{raw}}$
                  """ + fr"$Q_y^{{\text{{raw}}}}={raw_Qy%1}$")
    ax.set_title('Fractional Tune Footprint')
    
    ax.ticklabel_format(useOffset=False, style='plain')
    
    plt.tight_layout()
    
    if show==True:
        plt.show()

    return fig, ax