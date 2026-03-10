import numpy as np
import os
from pathlib import Path
import time
import h5py
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_p
import pandas as pd

import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

from cpymad.madx import Madx

@dataclass
class Simulation:
    folder: Path = Path('Lattice_Files/02_Aperture_Lattice/')
    threads: int = 1
    slices: int = 4
    #thick: bool = False
    n_macro_particles: int = 1000
    nemitt_x_0: float = 1e-6
    nemitt_y_0: float = 1e-6
    particle_type: str = 'proton'
    p0c: float = 0.37033168 * 1e9

    def __post_init__(self):
        self.line = self._build_line() 
        self.context = xo.ContextCpu(omp_num_threads=self.threads)

        self.line.set_particle_ref(self.particle_type, p0c=self.p0c)
        self.twiss = self.line.twiss(method='4d', freeze_longitudinal=True,)

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
            mode='normalized_transverse')
    

    def add_space_charge(self, beam_intensity, n_interactions):
        self.beam_intensity = beam_intensity
        self.n_interactions = n_interactions

        sigma_z_fake = 1e16 
        circumference = self.line.get_length()
        n_particles = self.beam_intensity/self.line.get_length()
        n_particles_fake = n_particles * np.sqrt(2*np.pi)*sigma_z_fake

        lprofile = xf.LongitudinalProfileQGaussian(
                number_of_particles= n_particles_fake,
                sigma_z= sigma_z_fake,
                z0= 0.,
                q_parameter= 1.0)


        xf.install_spacecharge_frozen(
            line=self.line,
            longitudinal_profile=lprofile,
            nemitt_x=self.nemitt_x_0, 
            nemitt_y=self.nemitt_y_0,
            sigma_z=sigma_z_fake,
            num_spacecharge_interactions=self.n_interactions,
            delta_rms=0)


    def run(self, n_turns):
        self.n_turns=n_turns

        self.line.build_tracker(_context=self.context)

        start_time=time.perf_counter()
        self.line.track(self.particles, num_turns=self.n_turns, with_progress=True, freeze_longitudinal=True, turn_by_turn_monitor=True)
        self.run_time = time.perf_counter() - start_time

        self.x_tbt = self.line.record_last_track.x
        self.px_tbt = self.line.record_last_track.px
        self.y_tbt = self.line.record_last_track.y
        self.py_tbt = self.line.record_last_track.py


    def save(self, filename='untitled'):

        df_twiss = self.twiss.to_pandas()
        df_twiss.to_hdf(f'data/{filename}.h5', key='twiss_data', mode='w', format='fixed')

        bigaussians = [e for e in self.line.elements if isinstance(e, xf.SpaceChargeBiGaussian)]

        sigma_x_list = []
        sigma_y_list = []
        mean_x_list = []
        mean_y_list = []
        for bigaussian in bigaussians:
            sigma_x = bigaussian.sigma_x
            sigma_y = bigaussian.sigma_y
            mean_x = bigaussian.mean_x
            mean_y = bigaussian.mean_y
            sigma_x_list.append(sigma_x)
            sigma_y_list.append(sigma_y)
            mean_x_list.append(mean_x)
            mean_y_list.append(mean_y)

        sigma_x = np.mean(np.array(sigma_x_list))
        sigma_y = np.mean(np.array(sigma_y_list))
        mean_x = np.mean(np.array(mean_x_list))
        mean_y = np.mean(np.array(mean_y_list))

        with h5py.File(f'data/{filename}.h5', 'a') as f:
            f.create_dataset('x_tbt', data=self.x_tbt, compression='gzip')
            f.create_dataset('y_tbt', data=self.y_tbt, compression='gzip')
            f.create_dataset('px_tbt', data=self.px_tbt, compression='gzip')
            f.create_dataset('py_tbt', data=self.py_tbt, compression='gzip')

            f.attrs['time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            f.attrs['threads'] = self.threads
            f.attrs['slices'] = self.slices
            #f.attrs['thick'] = self.thick

            f.attrs['gamma'] = self.particles.gamma0[0]
            f.attrs['beta'] = self.particles.beta0[0]
            
            f.attrs['n_macro_particles'] = self.n_macro_particles
            f.attrs['n_turns'] = self.n_turns
            f.attrs['nemitt_x_0'] = self.nemitt_x_0
            f.attrs['nemitt_y_0'] = self.nemitt_y_0

            f.attrs['qx'] = self.twiss.qx
            f.attrs['qy'] = self.twiss.qy
            f.attrs['dqx'] = self.twiss.dqx
            f.attrs['dqy'] = self.twiss.dqy

            f.attrs['sigma_x'] = sigma_x
            f.attrs['sigma_y'] = sigma_y
            f.attrs['mean_x'] = mean_x
            f.attrs['mean_y'] = mean_y

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



