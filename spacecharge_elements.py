import numpy as np
from scipy.constants import epsilon_0
k_e = 1/(4*np.pi*epsilon_0)

import xtrack as xt

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

