import os
from pathlib import Path
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e, m_p, epsilon_0
k_e = 1/(4*np.pi*epsilon_0)

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

from cpymad.madx import Madx

from spacecharge_elements import DirectSpaceChargeElement
from spacecharge_elements import IndirectSpaceChargeElement



def build_line(folder=Path('../Lattice_Files/02_aper_Lattice/'), 
               slices=4, thick=False, install_apertures=True):

    madx = Madx(stdout=False)
    madx.call(str(folder / 'ISIS.injected_beam'))
    madx.call(str(folder / 'ISIS.elements'))
    madx.call(str(folder / 'ISIS.strength'))
    madx.call(str(folder / 'ISIS.sequence'))
    madx.call(str(folder / 'ISIS.aperture'))
    madx.use('synchrotron')
    madx.command.select(flag='makethin', slice=slices, thick=thick)
    madx.command.makethin(sequence='synchrotron', style='teapot', 
                          makedipedge=True)

    line = xt.Line.from_madx_sequence(
                madx.sequence.synchrotron, 
                install_apertures=install_apertures
                )
    line.set_particle_ref('proton', p0c=0.37033168 * 1e9)
    return line


def add_dipole(line, strength, s=0, mode='normal'):
    s_list = s if hasattr(s, '__len__') else [s]
    if not hasattr(strength, '__len__'):
        strengths = [strength] * len(s_list)
    else:
        strengths = strength
        
    kicks = []
    if mode=='skew':
        for pos, st in zip(s_list, strengths):
            dipole = xt.Multipole(ksl=[st])
            name = f'dipole_at_{pos:.3f}'
            kicks.append(line.env.place(name, obj=dipole, at=pos))
    elif mode=='normal':
        for pos, st in zip(s_list, strengths):
            dipole = xt.Multipole(knl=[st])
            name = f'dipole_at_{pos:.3f}'
            kicks.append(line.env.place(name, obj=dipole, at=pos))
    line.insert(kicks)

    return line


def add_monitors(line, particles, n_monitors, n_turns):
    s_positions = np.linspace(0, line.get_length(), n_monitors, endpoint=False)
    n_particles = len(particles.particle_id) 

    monitors = []
    for i, s in enumerate(s_positions):
        mon = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                                  num_particles=n_particles)
        monitors.append(line.env.place(f'mon_{i}', obj=mon, at=s))

    line.insert(monitors)
    return line


def add_spacecharge(line, beam_intensity=1e13, n_interactions=100, 
                     nemitt=None, gemitt=None,
                     indirect=False, xfields=False,
                     pipe_enlargement=1):
    
    betagamma = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    if nemitt is None and gemitt is None:
        gemitt = 60e-6
    gemitt = [gemitt, gemitt] if isinstance(gemitt, (int, float)) else gemitt
    nemitt = [nemitt, nemitt] if isinstance(nemitt, (int, float)) else nemitt
    if nemitt is None:
        nemitt = [g * betagamma for g in gemitt]

    length = line.get_length()
    line_density = e * beam_intensity/length
    kick_length = length/n_interactions

    s_positions = np.linspace(0, length, n_interactions, endpoint=False)
    beam_df = _get_beam(line, s_positions, nemitt, betagamma)
    aper_df = _get_apers(line, s_positions, pipe_enlargement)

    beam_df['beam_intensity'] = beam_intensity
    beam_df['n_interactions'] = n_interactions
    beam_df['nemitt_x'] = nemitt[0]
    beam_df['nemitt_y'] = nemitt[1]
    aper_df['pipe_enlargement'] = pipe_enlargement

    if indirect:
        insertions = [
            line.env.place(
                f'sc_ele_{i}',                 
                obj=IndirectSpaceChargeElement(
                    element_length=kick_length, 
                    x0=beam.x0, y0=beam.y0, 
                    sigma_x=beam.sigma_x, sigma_y=beam.sigma_y, 
                    x_pipe=aper.x, y_pipe=aper.y,
                    x_length=aper.Lx, y_length=aper.Ly,
                    line_density=line_density
                ),
                at=s,
            )
            for i, (s, beam, aper) in enumerate(zip(s_positions, 
                                                    beam_df.itertuples(), 
                                                    aper_df.itertuples()))
        ]
        line.insert(insertions)

        line = _update_apertures(line, pipe_enlargement)

    elif not xfields:
        insertions = [
            line.env.place(
                f'sc_ele_{i}',
                obj=DirectSpaceChargeElement(
                    element_length=kick_length, 
                    x0=beam.x0, y0=beam.y0, 
                    sigma_x=beam.sigma_x, sigma_y=beam.sigma_y, 
                    line_density=line_density
                ),
                at=s,
            )
            for i, (s, beam) in enumerate(zip(s_positions, 
                                              beam_df.itertuples()))
        ]
        line.insert(insertions)

    else: # Xfields direct SC
        sigma_z_fake = 1e16 # Arbitrary must be >>length
        particle_line_density = beam_intensity/length
        n_particles_fake = particle_line_density*np.sqrt(2*np.pi)*sigma_z_fake

        lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles= n_particles_fake,
                sigma_z= sigma_z_fake
            )
        xf.install_spacecharge_frozen(
            line=line,
            longitudinal_profile=lprofile,
            nemitt_x=nemitt[0], 
            nemitt_y=nemitt[1],
            sigma_z=sigma_z_fake,
            num_spacecharge_interactions=n_interactions,
            delta_rms=0
            )

    return line, beam_df, aper_df


def save(line,
         particles,
         twiss,
         filename='untitled',
         aper_df=None,
         beam_df=None):

    twiss_df = twiss.to_pandas()
    twiss_df.to_hdf(f'{filename}.h5', key='twiss_df', 
                    mode='w', format='fixed')
    mon_mask = twiss_df['name'].str.contains('mon')

    twiss_df_at_mon = twiss_df[mon_mask]
    twiss_df_at_mon.to_hdf(f'{filename}.h5', key='twiss_df_at_mon', 
                    mode='a', format='fixed')

    if beam_df is not None:
        beam_df.to_hdf(f'{filename}.h5', key='beam_df', 
                    mode='a', format='fixed')

    if aper_df is not None:
        aper_df.to_hdf(f'{filename}.h5', key='aper_df', 
                           mode='a', format='fixed')
    
    x = _unfold(line, 'x')
    y = _unfold(line, 'y')
    px = _unfold(line, 'px')
    py = _unfold(line, 'py')

    survivor_mask = particles.state>0
    surviving_ids = particles.particle_id[survivor_mask]
    n_particles = len(particles.particle_id) 
    n_survivors = np.sum(survivor_mask)
    survivor_mask = np.zeros(n_particles, dtype=bool)
    survivor_mask[surviving_ids] = True

    n_monitors = len(twiss_df_at_mon)
    n_turns = len(x.T)//n_monitors

    with h5py.File(f'{filename}.h5', 'a') as f:
        f.create_dataset('x', data=x, compression='gzip')
        f.create_dataset('y', data=y, compression='gzip')
        f.create_dataset('px', data=px, compression='gzip')
        f.create_dataset('py', data=py, compression='gzip')

        f.create_dataset('survivor_mask', data=survivor_mask, 
                         compression='gzip')
        f.create_dataset('particle_ids', data=particles.particle_id, 
                         compression='gzip')

        f.attrs['gamma'] = line.particle_ref.gamma0
        f.attrs['beta'] = line.particle_ref.beta0
        f.attrs['p0c'] = line.particle_ref.p0c
        f.attrs['length'] = line.get_length()

        f.attrs['n_particles'] = n_particles
        f.attrs['n_turns'] = n_turns
        f.attrs['n_monitors'] = n_monitors

        f.attrs['qx'] = twiss.qx
        f.attrs['qy'] = twiss.qy
        f.attrs['dqx'] = twiss.dqx
        f.attrs['dqy'] = twiss.dqy
        
    print(f'File saved at {filename}')


########################################################################


def _get_beam(line, s_positions, nemitt, betagamma):
    tw = line.twiss(method='4d', freeze_longitudinal=True)
    indices = [np.argmin(np.abs(tw.s - s)) for s in s_positions]
    
    df_beam = pd.DataFrame({
        's': tw.s[indices],
        'x0': tw.x[indices],
        'y0': tw.y[indices],
        'betx': tw.betx[indices],
        'bety': tw.bety[indices],
        'sigma_x': np.sqrt(tw.betx[indices] * nemitt[0] / betagamma),
        'sigma_y': np.sqrt(tw.bety[indices] * nemitt[1] / betagamma),
    })
    
    return df_beam


def _get_apers(line, s_array, pipe_enlargement):
    tab = line.get_table().rows.match('.*aper')
    elements = [line.element_dict[n] for n in tab.name]
    
    df = pd.DataFrame({
        'name': tab.name,
        's': tab.s,
        'x':  [(a.max_x + a.min_x)/2 for a in elements],
        'Lx': [(a.max_x - a.min_x)*pipe_enlargement for a in elements],
        'y':  [(a.max_y + a.min_y)/2 for a in elements],
        'Ly': [(a.max_y - a.min_y)*pipe_enlargement for a in elements]
    })

    idx = np.clip(np.searchsorted(df.s.values, s_array), 0, len(df) - 1)
    
    return df.iloc[idx].reset_index(drop=True)


def _unfold(line, prop):
    mon_names = [name for name in line.element_names if name.startswith('mon_')]
    data_list = [getattr(line.element_dict[name], prop) for name in mon_names]
    stacked = np.stack(data_list)
    transposed = stacked.transpose(1, 2, 0)
    return np.array(transposed.reshape(transposed.shape[0], -1))

def _update_apertures(line, ratio):
    for name, element in line.element_dict.items():
        if isinstance(element, (xt.LimitRect)):
            if hasattr(element, 'min_x'): element.min_x *= ratio
            if hasattr(element, 'max_x'): element.max_x *= ratio
            
            if hasattr(element, 'min_y'): element.min_y *= ratio
            if hasattr(element, 'max_y'): element.max_y *= ratio
    return line
