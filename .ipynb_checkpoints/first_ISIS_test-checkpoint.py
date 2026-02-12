#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import xobjects as xo
from cpymad.madx import Madx

folder = Path('Lattice_Files/00_Simplified_Lattice/')

madx = Madx(stdout=True)
madx.call(str(folder / 'ISIS.injected_beam'))
madx.call(str(folder / 'ISIS.elements'))
madx.call(str(folder / 'ISIS.strength'))
madx.call(str(folder / 'ISIS.sequence'))
madx.use('synchrotron')


line = xt.Line.from_madx_sequence(madx.sequence.synchrotron)

line.set_particle_ref('proton', kinetic_energy0=70e6)

context = xo.ContextCpu()                         # For CPU (single thread)
line.build_tracker()

n_particles = int(1e3)
particles = xt.Particles(
    x=np.random.normal(0, 1e-6, n_particles),
    px=np.random.normal(0, 1e-6, n_particles),
    y=np.random.normal(0, 1e-6, n_particles),
    py=np.random.normal(0, 1e-6, n_particles),
    zeta=np.random.normal(0, 1e-5, n_particles),
    delta=np.random.normal(0, 1e-5, n_particles)
)

n_turns = 100
line.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)


fig, axs = plt.subplots(2)
axs[0].scatter(particles.x, particles.px)
axs[0].set_xlabel("x")
axs[0].set_ylabel("px")
axs[1].scatter(particles.y, particles.py)
axs[1].set_xlabel("y")
axs[1].set_ylabel("py")
plt.show()
