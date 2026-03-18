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

from pipefields.rectangle import get_field

class SpaceChargeElement(xt.BeamElement):
    _xofields = {
        'element_length': 'float64',
        'sigma_x':      'float64',
        'sigma_y':      'float64',
        'min_x':        'float64',
        'max_x':        'float64',
        'min_y':        'float64',
        'max_y':        'float64',
        'line_density': 'float64',
    }

    iscollective = True

    def __init__(self, **kwargs):
        super().__init__() 
        
        self.element_length = kwargs.get('element_length', 0)
        self.sigma_x = kwargs.get('sigma_x', 0)
        self.sigma_y = kwargs.get('sigma_y', 0)
        self.min_x = kwargs.get('min_x', 0)
        self.max_x = kwargs.get('max_x', 0)
        self.min_y = kwargs.get('min_y', 0)
        self.max_y = kwargs.get('max_y', 0)
        self.line_density = kwargs.get('line_density', 0)

    def track(self, particles):
        x0 = self.max_x + self.min_x
        y0 = self.max_y + self.min_y
        Lx = self.max_x - self.min_x
        Ly = self.max_y - self.min_y

        Ex, Ey = get_field(particles.x, particles.y, 
                           x0=x0, y0=y0, 
                           sx=self.sigma_x, sy=self.sigma_y, 
                           Lx=Lx, Ly=Ly)
        
        beta0 = particles.beta0[0]
        gamma0 = particles.gamma0[0]
        p0c = particles.p0c[0]
        
        coef = (k_e * self.line_density) / (p0c * beta0 * gamma0)

        particles.px += coef * self.element_length * Ex
        particles.py += coef * self.element_length * Ey


@dataclass
class Simulation:
    folder: Path = Path('Lattice_Files/02_Aperture_Lattice/')
    threads: int = 1
    slices: int = 4
    n_macro_particles: int = 1000
    nemitt_x_0: float = 1e-6
    nemitt_y_0: float = 1e-6
    particle_type: str = 'proton'
    p0c: float = 0.37033168 * 1e9

    def __post_init__(self):
        self.line = self._build_line() 
        self.context = xo.ContextCpu(omp_num_threads=self.threads)

        self.line.set_particle_ref(self.particle_type, p0c=self.p0c)
        self.beta0=self.line.particle_ref.beta0
        self.gamma0=self.line.particle_ref.gamma0
        self.energy0=self.line.particle_ref.energy0

        self.twiss = self.line.twiss(method='4d', freeze_longitudinal=True,)
        self.length = self.line.get_length()


    

    def insert_dipole_error(self, strength, s=0):
        dipole = xt.Multipole(ksl=[strength])

        if hasattr(s, '__len__'):
            kicks = []
            for pos in s:
                kicks.append(self.line.env.place(f'dipole_error_at_{pos}', obj=dipole, at=pos))
            self.line.insert(kicks)
        else:
            kick = self.line.env.place(f'dipole_error_at_{s}', obj=dipole, at=s)
            self.line.insert(kick)


    def add_space_charge(self, beam_intensity, n_interactions):
        self.beam_intensity = beam_intensity
        self.n_interactions = n_interactions

        line_density = e * self.beam_intensity/self.length
        kick_length = self.length/n_interactions

        s_positions = np.linspace(0, self.length, n_interactions, endpoint=False)
        sigma_x_list, sigma_y_list = self._get_sigmas(s_positions, self.nemitt_x_0, self.nemitt_y_0)
        min_x_list, max_x_list, min_y_list, max_y_list = self._get_apertures(s_positions)

        insertions = []
        for i, s in enumerate(s_positions):
            sc_ele = SpaceChargeElement(element_length=kick_length, 
                                        sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[i], 
                                        min_x=min_x_list[i], max_x=max_x_list[i], 
                                        min_y=min_y_list[i], max_y=max_y_list[i], 
                                        line_density=line_density)
            insertions.append(self.line.env.place(f'sc_ele_{i}', obj=sc_ele, at=s))

        self.sigma_x_list = sigma_x_list
        self.sigma_y_list = sigma_y_list
        self.mean_x_list = min_x_list + max_x_list
        self.mean_y_list = min_y_list + max_y_list

        self.line.insert(insertions)
        self.line.build_tracker()


    def _get_sigmas(self, s_positions, nemitt_x, nemitt_y):
        line_copy = self.line.copy(shallow=True)
        line_copy.cut_at_s(s_positions)
        line_copy.build_tracker()

        tw = line_copy.twiss(method='4d', freeze_longitudinal=True)
        indices = [np.argmin(np.abs(tw.s - s)) for s in s_positions]
        
        betx = tw.betx[indices]
        bety = tw.bety[indices]

        sigmas_x = np.sqrt(betx * nemitt_x / (self.beta0*self.gamma0))
        sigmas_y = np.sqrt(bety * nemitt_y / (self.beta0*self.gamma0))
        
        return sigmas_x, sigmas_y

    def _get_apertures(self, s_array):
        element_table = self.line.get_table()
        aperture_table = element_table.rows.match('.*aper')

        distances = np.abs(np.subtract.outer(s_array, aperture_table.s))
        closest_indices = np.argmin(distances, axis=1)
        active_aper_names = aperture_table.name[closest_indices]
        
        results = []
        for name in active_aper_names:
            aper_obj = self.line.element_dict[name]
            results.append((aper_obj.min_x, aper_obj.max_x, aper_obj.min_y, aper_obj.max_y))
        
        return np.array(results).T


    def add_monitors(self, n_monitors, n_turns):
        s_positions = np.linspace(0, self.length, n_monitors, endpoint=False)
        
        monitors = []
        monitor_names = []
        for i, s in enumerate(s_positions):
            try:
                mon = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                                          num_particles=self.n_macro_particles)
            except:
                mon = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                                          num_particles=1)

            monitors.append(self.line.env.place(f'mon_{i}', obj=mon, at=s))
            monitor_names.append(f'mon_{i}')
        self.line.insert(monitors)
        self.monitor_names = monitor_names
        self.n_monitors = n_monitors
        self.line.build_tracker()
        self.twiss = self.line.twiss(method='4d', freeze_longitudinal=True,)


    def run(self, n_turns):
        if self.n_macro_particles=='co':
            self.particles = self.line.build_particles(
                    x=self.twiss.x[0], 
                    y=self.twiss.y[0], 
                    px=self.twiss.px[0], 
                    py=self.twiss.py[0],
                    method='4d', freeze_longitudinal=True,
            )
        else:
            self.particles = self.line.build_particles(
                num_particles=self.n_macro_particles,
                x_norm=np.random.normal(size=self.n_macro_particles),
                y_norm=np.random.normal(size=self.n_macro_particles),
                px_norm=np.random.normal(size=self.n_macro_particles),
                py_norm=np.random.normal(size=self.n_macro_particles),
                nemitt_x=self.nemitt_x_0,
                nemitt_y=self.nemitt_y_0,
                method='4d',
                freeze_longitudinal=True,
                mode='normalized_transverse'
            )


        self.n_turns=n_turns
        self.line.build_tracker(_context=self.context)

        start_time=time.perf_counter()
        self.line.track(self.particles, num_turns=n_turns, time=True)
        self.run_time = time.perf_counter() - start_time

        
    def _unfold(self, prop):
        data_list = [getattr(self.line.element_dict[name], prop) for name in self.monitor_names]

        stacked = np.stack(data_list)
        transposed = stacked.transpose(1, 2, 0)
        unfolded = transposed.reshape(transposed.shape[0], -1)
        
        return unfolded


    def save(self, filename='untitled'):
        df_twiss = self.twiss.to_pandas()
        df_twiss.to_hdf(f'data/{filename}.h5', 
                        key='twiss_data', 
                        mode='w', 
                        format='fixed')

        self.x = np.array(self._unfold('x')).T
        self.y = np.array(self._unfold('y')).T
        self.px = np.array(self._unfold('px')).T
        self.py = np.array(self._unfold('py')).T

        with h5py.File(f'data/{filename}.h5', 'a') as f:
            f.create_dataset('x', data=self.x, compression='gzip')
            f.create_dataset('y', data=self.y, compression='gzip')
            f.create_dataset('px', data=self.px, compression='gzip')
            f.create_dataset('py', data=self.py, compression='gzip')

            f.attrs['time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            f.attrs['threads'] = self.threads
            f.attrs['slices'] = self.slices

            f.attrs['gamma'] = self.gamma0
            f.attrs['beta'] = self.beta0
            
            f.attrs['n_macro_particles'] = self.n_macro_particles
            f.attrs['n_turns'] = self.n_turns
            f.attrs['nemitt_x_0'] = self.nemitt_x_0
            f.attrs['nemitt_y_0'] = self.nemitt_y_0

            f.attrs['qx'] = self.twiss.qx
            f.attrs['qy'] = self.twiss.qy
            f.attrs['dqx'] = self.twiss.dqx
            f.attrs['dqy'] = self.twiss.dqy
            
            try:
                f.create_dataset('sigma_x', data=self.sigma_x_list, compression='gzip')
                f.create_dataset('sigma_y', data=self.sigma_y_list, compression='gzip')
                f.create_dataset('mean_x', data=self.mean_x_list, compression='gzip')
                f.create_dataset('mean_y', data=self.mean_y_list, compression='gzip')

                f.attrs['n_interactions'] = self.n_interactions
                f.attrs['beam_intensity'] = self.beam_intensity
            except:
                pass
            f.attrs['n_monitors'] = self.n_monitors
            f.attrs['monitor_names'] = self.monitor_names

        print(f'File saved at data/{filename}')


    def _build_line(self):
        madx = Madx(stdout=False)
        madx.call(str(self.folder / 'ISIS.injected_beam'))
        madx.call(str(self.folder / 'ISIS.elements'))
        madx.call(str(self.folder / 'ISIS.strength'))
        madx.call(str(self.folder / 'ISIS.sequence'))
        madx.call(str(self.folder / 'ISIS.aperture'))
        madx.use('synchrotron')
        madx.command.select(flag='makethin', slice=self.slices, thick=False)
        madx.command.makethin(sequence='synchrotron', style='teapot', makedipedge=True)

        line = xt.Line.from_madx_sequence(madx.sequence.synchrotron, install_apertures=True)

        return line


"""
Next steps
----------

- Calculate the closed orbit for the line including dipole kicks but not sc 
- Centre s.c. on closed orbit
- Can removed co tracking instead add closed orbit as new method??
- Verify dipoles kick closed orbit as expected
- 
"""
