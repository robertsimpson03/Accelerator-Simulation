import numpy as np
import os
from pathlib import Path
import time
import h5py
from datetime import datetime

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
class simulation:
    folder: str = 02_Aperture_Lattice
    threads: int = 1
    slices: int = 4
    n_macro_particles: int = 1000
    nemitt_x_0: float = 1e-6
    nemitt_y_0: float = 1e-6
    particle_type: str = 'proton'
    p0c: float = 0.37033168 * 1e9

    def __post_init__():
        self.line = self._build_line() 
        self.context = xo.ContextCpu(omp_num_threads=self.threads)

        self.line.set_particle_ref(self.particle_type, p0c=self.p0c)
        self.twiss = self.line.twiss(method='4d', freeze_longitudinal=True,)

        self.particles = line.build_particles(
            num_particles=n_particles,
            x_norm=np.random.normal(size=self.n_macro_particles),
            y_norm=np.random.normal(size=self.n_macro_particles),
            px_norm=np.random.normal(size=self.n_macro_particles),
            py_norm=np.random.normal(size=self.n_macro_particles),
            nemitt_x=self.nemitt_x_0,
            nemitt_y=self.nemitt_y_0,
            method='4d',
            freeze_longitudinal=True,
            mode='normalized_transverse',
        )
    

    def add_space_charge(self, beam_intensity, n_interactions):
        self.beam_intensity = beam_intensity
        self.n_interactions = n_interactions

        lprofile_data = {
            '__class__': 'LongitudinalProfileCoasting',
            'beam_line_density': self.beam_intensity / self.line.get_length()
        }

        xf.install_spacecharge_frozen(line=self.line,
                           longitudinal_profile=lprofile_data,
                           nemitt_x=self.nemitt_x_0, nemitt_y=self.nemitt_y_0,
                           sigma_z=0,
                           num_spacecharge_interactions=self.n_interactions,
                           delta_rms=0
                   )
        #xf.SpaceChargeBiGaussian(longitudinal_profile=lprofile_data)


    def run(self, n_turns):
        self.n_turns=n_turns

        self.line.build_tracker(_context=self.context)
        n_turns = 200

        start_time=time.perf_counter()
        self.line.track(particles, num_turns=self.n_turns, with_progress=True, freeze_longitudinal=True, turn_by_turn_monitor=True)
        self.run_time = time.perf_counter() - start_time


    def save(name='untitled'):
        pass


    def _build_Line(self):
        madx = Madx(stdout=False)
        madx.call(str(self.folder / 'ISIS.injected_beam'))
        madx.call(str(self.folder / 'ISIS.elements'))
        madx.call(str(self.folder / 'ISIS.strength'))
        madx.call(str(self.folder / 'ISIS.sequence'))
        madx.call(str(self.folder / 'ISIS.aperture'))
        madx.use('synchrotron')
        madx.command.select(flag='makethin', slice=self.slices, thick=self.thick)
        madx.command.makethin(sequence='synchrotron', style='teapot', makedipedge=True)

        line = xt.Line.from_madx_sequence(madx.sequence.synchrotron, install_apertures=True)

        return line



