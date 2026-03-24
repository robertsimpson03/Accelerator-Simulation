import os
from pathlib import Path
import time
import h5py

from datetime import datetime
from dataclasses import dataclass

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

from pipefields.rectangle import get_field as rectangular_field
from pipefields.freespace import get_field as free_field

class IndirectSpaceChargeElement(xt.BeamElement):
    _xofields = {
        'element_length': 'float64',
        'x0':      'float64',
        'y0':      'float64',
        'sigma_x':      'float64',
        'sigma_y':      'float64',
        'x_pipe':      'float64',
        'y_pipe':      'float64',
        'x_length':      'float64',
        'y_length':      'float64',
        'line_density': 'float64',
    }
    iscollective = True

    def __init__(self, **kwargs):
        super().__init__() 
        
        self.element_length = kwargs.get('element_length', 0)
        self.x0 = kwargs.get('x0', 0)
        self.y0 = kwargs.get('y0', 0)
        self.sigma_x = kwargs.get('sigma_x', 0)
        self.sigma_y = kwargs.get('sigma_y', 0)
        self.x_pipe = kwargs.get('x_pipe', 0)
        self.y_pipe = kwargs.get('y_pipe', 0)
        self.x_length = kwargs.get('x_length', 0)
        self.y_length = kwargs.get('y_length', 0)
        self.line_density = kwargs.get('line_density', 0)

    def track(self, particles):
        Ex, Ey = rectangular_field(
                particles.x-self.x_pipe, particles.y-self.y_pipe, 
                x0=self.x0-self.x_pipe, y0=self.y0-self.y_pipe, 
                sx=self.sigma_x, sy=self.sigma_y,
                Lx=self.x_length, Ly=self.y_length)
        
        coef = (k_e * self.line_density * self.element_length
                /(particles.energy0[0] * (particles.beta0[0]
                                         *particles.gamma0[0])**2))

        particles.px += coef * Ex
        particles.py += coef * Ey


class DirectSpaceChargeElement(xt.BeamElement):
    _xofields = {
        'element_length': 'float64',
        'x0':      'float64',
        'y0':      'float64',
        'sigma_x':      'float64',
        'sigma_y':      'float64',
        'line_density': 'float64',
    }
    iscollective = True

    def __init__(self, **kwargs):
        super().__init__() 
        
        self.element_length = kwargs.get('element_length', 0)
        self.x0 = kwargs.get('x0', 0)
        self.y0 = kwargs.get('y0', 0)
        self.sigma_x = kwargs.get('sigma_x', 0)
        self.sigma_y = kwargs.get('sigma_y', 0)
        self.line_density = kwargs.get('line_density', 0)

    def track(self, particles):
        Ex, Ey = free_field(
                particles.x, particles.y, 
                x0=self.x0, y0=self.y0, 
                sx=self.sigma_x, sy=self.sigma_y)
        
        coef = (k_e * self.line_density * self.element_length
                /(particles.energy0[0] * (particles.beta0[0]
                                         *particles.gamma0[0])**2))

        particles.px += coef * Ex
        particles.py += coef * Ey


@dataclass
class Simulation:
    folder: Path = Path('Lattice_Files/02_Aperture_Lattice/')
    threads: int = 0
    slices: int = 4
    particle_type: str = 'proton'
    p0c: float = 0.37033168 * 1e9


    def __post_init__(self):
        self.line = _build_line(self.folder, self.slices) 
        self.context = xo.ContextCpu(omp_num_threads=self.threads)
        self.length = self.line.get_length()

        self.line.set_particle_ref(self.particle_type, p0c=self.p0c)

    def add_dipole(self, strength=1e-4, s=0, mode='skew'):
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
                kicks.append(self.line.env.place(name, obj=dipole, at=pos))
        elif mode=='normal':
            for pos, st in zip(s_list, strengths):
                dipole = xt.Multipole(knl=[st])
                name = f'dipole_at_{pos:.3f}'
                kicks.append(self.line.env.place(name, obj=dipole, at=pos))

        
        self.line.insert(kicks)
        
    def build_particles(self, n_particles=100, nemitt_x_0=1e-6, nemitt_y_0=1e-6, co=False,
                        offset_x=0, offset_y=0, offset_px=0, offset_py=0):
        if co==False:
            self.particles = _build_particles(
                    self.line, n_particles, nemitt_x_0, nemitt_y_0,
                    offset_x, offset_y, offset_px, offset_py)
        else:
            self.particles = _build_co_particle(
                    self.line, offset_x, offset_y, offset_px, offset_py)
            n_particles=1
        self.n_particles = n_particles
        self.nemitt_x_0 = nemitt_x_0
        self.nemitt_y_0 = nemitt_y_0

    def set_up(self, n_turns=100, n_monitors=100):
        self.line = _add_monitors(self.line, n_monitors, n_turns, self.n_particles)
        self.twiss = self.line.twiss(method='4d', freeze_longitudinal=True,)
        self.n_turns = n_turns
        self.n_monitors = n_monitors

    def run(self, spacecharge=False, beam_intensity=1e9, 
            n_interactions=100, indirect_spacecharge=False,
            xfields=False, pipe_enlargement=1):
        if spacecharge==True:
            (
                self.line, self.x0, self.y0, self.sigma_x, self.sigma_y, 
                self.x_pipe, self.y_pipe, self.x_length, self.y_length
            ) = _add_spacecharge(self.line, 
                                 beam_intensity, n_interactions, 
                                 self.nemitt_x_0, self.nemitt_y_0,
                                 indirect_spacecharge, xfields,
                                 pipe_enlargement)
            self.beam_intensity = beam_intensity
            self.n_interactions = n_interactions

        self.line.build_tracker(_context=self.context)

        start_time=time.perf_counter()
        self.line.track(self.particles, num_turns=self.n_turns, time=True)
        self.run_time = time.perf_counter() - start_time


    def save(self, filename='untitled'):
        df_twiss = self.twiss.to_pandas()
        df_twiss.to_hdf(f'{filename}.h5', key='twiss_data', 
                        mode='w', format='fixed')

        x = _unfold(self.line, 'x')
        y = _unfold(self.line, 'y')
        px = _unfold(self.line, 'px')
        py = _unfold(self.line, 'py')

        with h5py.File(f'{filename}.h5', 'a') as f:
            f.create_dataset('x', data=x, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.create_dataset('px', data=px, compression='gzip')
            f.create_dataset('py', data=py, compression='gzip')

            f.create_dataset('survivor_mask', data=self.particles.state>0, compression='gzip')

            f.attrs['time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            f.attrs['runtime'] = self.run_time
            f.attrs['threads'] = self.threads
            f.attrs['slices'] = self.slices

            f.attrs['gamma'] = self.line.particle_ref.gamma0
            f.attrs['beta'] = self.line.particle_ref.beta0
            f.attrs['p0c'] = self.line.particle_ref.p0c
            f.attrs['length'] = self.line.get_length()

            f.attrs['n_particles'] = self.n_particles
            f.attrs['n_turns'] = self.n_turns
            f.attrs['n_monitors'] = self.n_monitors
            f.attrs['nemitt_x_0'] = self.nemitt_x_0
            f.attrs['nemitt_y_0'] = self.nemitt_y_0

            f.attrs['qx'] = self.twiss.qx
            f.attrs['qy'] = self.twiss.qy
            f.attrs['dqx'] = self.twiss.dqx
            f.attrs['dqy'] = self.twiss.dqy
            
            try:
                f.attrs['n_interactions'] = self.n_interactions
                f.attrs['beam_intensity'] = self.beam_intensity

                f.create_dataset('x0', data=self.x0, compression='gzip')
                f.create_dataset('y0', data=self.y0, compression='gzip')
                f.create_dataset('sigma_x', data=self.sigma_x, compression='gzip')
                f.create_dataset('sigma_y', data=self.sigma_y, compression='gzip')
                f.create_dataset('x_pipe', data=self.x_pipe, compression='gzip')
                f.create_dataset('y_pipe', data=self.y_pipe, compression='gzip')
                f.create_dataset('x_length', data=self.x_length, compression='gzip')
                f.create_dataset('y_length', data=self.y_length, compression='gzip')
            except:
                pass

        print(f'File saved at {filename}')

# --- Build line ---
def _build_line(folder, slices):
        madx = Madx(stdout=False)
        madx.call(str(folder / 'ISIS.injected_beam'))
        madx.call(str(folder / 'ISIS.elements'))
        madx.call(str(folder / 'ISIS.strength'))
        madx.call(str(folder / 'ISIS.sequence'))
        madx.call(str(folder / 'ISIS.aperture'))
        madx.use('synchrotron')
        madx.command.select(flag='makethin', slice=slices, thick=False)
        madx.command.makethin(sequence='synchrotron', 
                              style='teapot', makedipedge=True)

        return xt.Line.from_madx_sequence(madx.sequence.synchrotron, 
                                          install_apertures=True)

# --- Build particles ---

def _build_particles(line, n_particles, nemitt_x_0, nemitt_y_0, 
                     offset_x, offset_y, offset_px, offset_py):
    tw = line.twiss(method='4d', freeze_longitudinal=True)
    particles = line.build_particles(
        x_norm=np.random.normal(size=n_particles),
        y_norm=np.random.normal(size=n_particles),
        px_norm=np.random.normal(size=n_particles),
        py_norm=np.random.normal(size=n_particles),
        nemitt_x=nemitt_x_0,
        nemitt_y=nemitt_y_0,
        method='4d',
        freeze_longitudinal=True,
        mode='normalized_transverse')
    particles.x += (tw.x[0] + offset_x - np.mean(particles.x))
    particles.y += (tw.y[0] + offset_y - np.mean(particles.y))
    particles.px += (tw.px[0] + offset_px - np.mean(particles.px))
    particles.py += (tw.py[0] + offset_py - np.mean(particles.py))

    return particles

def _build_co_particle(line, offset_x, offset_y, offset_px, offset_py):
    tw = line.twiss(method='4d', freeze_longitudinal=True)
    particle = line.build_particles(
        x=tw.x[0] + offset_x, 
        y=tw.y[0] + offset_y,
        px=tw.px[0] + offset_px, 
        py=tw.py[0] + offset_py)
    return particle

# --- Add monitors ---

def _add_monitors(line, n_monitors, n_turns, n_particles):
    s_positions = np.linspace(0, line.get_length(), n_monitors, endpoint=False)
    
    monitors = []
    for i, s in enumerate(s_positions):
        mon = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                                  num_particles=n_particles)
        monitors.append(line.env.place(f'mon_{i}', obj=mon, at=s))

    line.insert(monitors)
    return line

# --- Add spacecharge ---

def _add_spacecharge(line, beam_intensity, n_interactions, 
                     nemitt_x_0, nemitt_y_0, 
                     indirect_spacecharge, xfields,
                     pipe_enlargement):
    length = line.get_length()
    line_density = e * beam_intensity/length
    kick_length = length/n_interactions

    s_positions = np.linspace(0, length, n_interactions, endpoint=False)
    x0, y0, sigma_x, sigma_y = _get_beam(line, s_positions, 
                                         nemitt_x_0, nemitt_y_0)
    min_x, max_x, min_y, max_y = _get_apertures(line, s_positions)
    x_pipe = max_x + min_x
    y_pipe = max_y + min_y
    x_length = pipe_enlargement * (max_x - min_x)
    y_length = pipe_enlargement * (max_y - min_y)

    if indirect_spacecharge==True: # My rectangular indirect sc
        insertions = []
        for i, s in enumerate(s_positions):
            sc_ele = DirectSpaceChargeElement(
                    element_length=kick_length, 
                    x0=x0[i], y0=y0[i], 
                    sigma_x=sigma_x[i], sigma_y=sigma_y[i], 
                    line_density=line_density)
            insertions.append(line.env.place(f'sc_ele_{i}', obj=sc_ele, at=s))
        line.insert(insertions)

    elif xfields==False: # My direct SC
        insertions = []
        for i, s in enumerate(s_positions):
            sc_ele = DirectSpaceChargeElement(
                    element_length=kick_length, 
                    x0=x0[i], y0=y0[i], 
                    sigma_x=sigma_x[i], sigma_y=sigma_y[i], 
                    line_density=line_density)
            insertions.append(line.env.place(f'sc_ele_{i}', obj=sc_ele, at=s))
        line.insert(insertions)

    else: # Xfields direct SC
        sigma_z_fake = 1e16 
        circumference = line.get_length()
        n_particles = beam_intensity/line.get_length()
        n_particles_fake = n_particles * np.sqrt(2*np.pi)*sigma_z_fake

        lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles= n_particles_fake,
                sigma_z= sigma_z_fake,
                z0= 0.,
                q_parameter= 1.0)
        xf.install_spacecharge_frozen(
            line=line,
            longitudinal_profile=lprofile,
            nemitt_x=nemitt_x_0, 
            nemitt_y=nemitt_y_0,
            sigma_z=sigma_z_fake,
            num_spacecharge_interactions=n_interactions,
            delta_rms=0)

    return line, x0, y0, sigma_x, sigma_y, x_pipe, y_pipe, x_length, y_length

# --- Get Beam parameters ---

def _get_beam(line, s_positions, nemitt_x, nemitt_y):
    line_copy = line.copy(shallow=True)
    line_copy.cut_at_s(s_positions)
    line_copy.build_tracker()

    tw = line_copy.twiss(method='4d', freeze_longitudinal=True)
    indices = [np.argmin(np.abs(tw.s - s)) for s in s_positions]
    
    betx = tw.betx[indices]
    bety = tw.bety[indices]

    sigmas_x = np.sqrt(betx * nemitt_x 
                       /(line.particle_ref.beta0*line.particle_ref.gamma0))
    sigmas_y = np.sqrt(bety * nemitt_y 
                       /(line.particle_ref.beta0*line.particle_ref.gamma0))

    x0 = tw.x[indices] 
    y0 = tw.y[indices]
    
    return x0, y0, sigmas_x, sigmas_y


def _get_apertures(line, s_array):
    element_table = line.get_table()
    aperture_table = element_table.rows.match('.*aper')

    distances = np.abs(np.subtract.outer(s_array, aperture_table.s))
    closest_indices = np.argmin(distances, axis=1)
    active_aper_names = aperture_table.name[closest_indices]
    
    results = []
    for name in active_aper_names:
        aper_obj = line.element_dict[name]
        results.append((aper_obj.min_x, aper_obj.max_x, 
                        aper_obj.min_y, aper_obj.max_y))
    
    return np.array(results).T


def _unfold(line, prop):
    mon_names = [name for name in line.element_names if name.startswith('mon_')]
    data_list = [getattr(line.element_dict[name], prop) for name in mon_names]
    stacked = np.stack(data_list)
    transposed = stacked.transpose(1, 2, 0)
    return np.array(transposed.reshape(transposed.shape[0], -1))



