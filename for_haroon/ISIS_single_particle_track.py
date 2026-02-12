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

madx.use('sp_tune_m1')
madx.command.select(flag='makethin', slice=4, thick=False)
madx.command.makethin(sequence='sp_tune_m1', style='teapot', makedipedge=True)
madx.use('sp_tune_m1')

context = xo.ContextCpu()

line = xt.Line.from_madx_sequence(madx.sequence.sp_tune_m1)
line.build_tracker()

n_particles = 100
particles = xt.Particles()
    x=np.random.normal(0, 1e-6, n_particles),
    px=np.random.normal(0, 1e-6, n_particles),
    y=np.random.normal(0, 1e-6, n_particles),
    py=np.random.normal(0, 1e-6, n_particles),
    zeta=np.random.normal(0, 1e-5, n_particles),
    delta=np.random.normal(0, 1e-5, n_particles)
)

n_turns = 100
line.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)

x_tbt = line.record_last_track.x
px_tbt = line.record_last_track.px
y_tbt = line.record_last_track.y
py_tbt = line.record_last_track.py
zeta_tbt = line.record_last_track.zeta
delta_tbt = line.record_last_track.delta

fig, axs = plt.subplots(2)
for particle in range(n_particles):
    axs[0].plot(x_tbt[:, particle])
    axs[1].plot(px_tbt[:, particle])

plt.show()
