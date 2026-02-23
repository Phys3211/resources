# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:53:31 2026

@author: tomke
"""

import numpy as np

def beeman_step(r, v, a_prev, a_curr, force_func, dt, m=1.0):
    """
    Single Beeman step.
    
    Parameters
    ----------
    r, v       : current position and velocity
    a_prev     : acceleration at t - dt
    a_curr     : acceleration at t
    force_func : callable F(r) -> force
    dt         : timestep
    m          : mass
    
    Returns
    -------
    r_new, v_new, a_curr (which becomes a_prev next step), a_new
    """
    # Position update (same order as Verlet)
    r_new = r + v*dt + (dt**2 / 6.0) * (4*a_curr - a_prev)
    
    # New acceleration from updated position
    a_new = force_func(r_new) / m
    
    # Velocity update — 3rd order accurate
    v_new = v + (dt / 6.0) * (2*a_new + 5*a_curr - a_prev)
    
    return r_new, v_new, a_curr, a_new


def simulate_beeman(r0, v0, force_func, dt, n_steps, m=1.0):
    """Integrate equations of motion using Beeman's method."""
    r = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    
    r[0], v[0] = r0, v0
    a0 = force_func(r0) / m
    
    # Bootstrap: one Velocity Verlet step to get a(-dt) equivalent
    # Use a_prev = a0 - (da/dt)*dt ≈ a0 as a simple approximation,
    # or run one VV step backward
    # Here we just use VV for the first step:
    r[1] = r0 + v0*dt + 0.5*a0*dt**2
    a1 = force_func(r[1]) / m
    v[1] = v0 + 0.5*(a0 + a1)*dt
    
    a_prev, a_curr = a0, a1
    
    for i in range(1, n_steps):
        r[i+1], v[i+1], a_prev, a_curr = beeman_step(
            r[i], v[i], a_prev, a_curr, force_func, dt, m
        )
    
    return r, v