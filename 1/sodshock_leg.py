import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

class Particle():
    def __init__(self, x=[0, 0, 0], rho=1, v=[0, 0, 0], e=0, p=1, m=0.001875):
        self.x = x
        self.v = v
        self.e = e
        self.rho = rho
        self.p = p
        self.m = m

    def __str__(self):
        return f"x: {self.x}, v: {self.v}, e: {self.e}, rho: {self.rho}, p: {self.p}, m: {self.m}"

class Particles():
    def __init__(self, particles=[], r=[]):
        self.particles = particles
        if r.size != 0:
            for r_val in r:
                if r_val[0] <= 0:
                    self.particles.append(Particle(x=r_val, rho = 1, e = 2.5, p = 1))
                else:
                    self.particles.append(Particle(x=r_val, rho = 0.25, e = 1.795, p = 0.1795))
        
    def get_pressures(self):
        return np.array([prt.p for prt in self.particles])

    def get_densities(self):
        return np.array([prt.rho for prt in self.particles])

    def get_velocities(self):
        return np.array([prt.v for prt in self.particles])

    def get_masses(self):
        return np.array([prt.m for prt in self.particles])

# Coordinates of particles
r = np.concatenate((np.linspace(-0.6, 0, 320), np.linspace(0, 0.6, 80)))
r = np.stack((r, np.zeros_like(r), np.zeros_like(r)), axis=1)  # shape (N, 3)
particles = Particles(r=r)

r_i = np.expand_dims(r, axis=1)
r_j = np.expand_dims(r, axis=0)

# Distance between particles in x, y, z direction
dr = r_i - r_j
# Absolute 3D distance between particles
r_abs = np.linalg.norm(dr, axis=2)
R = r_abs / constants.h

print("r:", r.shape)
print("r_i:", r_i.shape)
print("r_j:", r_j.shape)
print("dr:", dr.shape)
print("r_abs:", r_abs.shape)
print("R:", R.shape)

def set_w(alpha, R, r_abs):
    w_ij = np.zeros_like(R)
    mask_nearest = (R >= 0) & (R < 1)
    mask_near = (R >= 1) & (R < 2)
    w_ij[mask_nearest] = alpha * ((2/3) - R[mask_nearest]**2 + 0.5 * R[mask_nearest]**3)
    w_ij[mask_near] = alpha * (1/6) * (2 - R[mask_near])**3

    R = np.repeat(R[..., np.newaxis], 3, axis=2)
    r_abs = np.repeat(r_abs[..., np.newaxis], 3, axis=2)

    w_ij_deriv = np.zeros_like(R)
    mask_nearest = (R >= 0) & (R < 1)
    mask_near = (R >= 1) & (R < 2)
    w_ij_deriv[mask_nearest] = alpha * (-2 + (1.5)*R[mask_nearest]) * (dr[mask_nearest]/(constants.h**2))
    w_ij_deriv[mask_near] = -alpha * 0.5 * (2 - R[mask_near])**2 * (dr[mask_near]/(constants.h*r_abs[mask_near]))

    return w_ij, w_ij_deriv

w_ij, w_ij_deriv = set_w(alpha=1/constants.h, R=R, r_abs=r_abs)

print("w_ij shape:", w_ij.shape)
print("w_ij_deriv shape:", w_ij_deriv.shape)


dt = 0.005
n_timesteps = 40


def sph():
    m_i = particles.get_masses()
    m_j = np.expand_dims(m_i, axis=0)
    p_i = particles.get_pressures()
    p_j = np.expand_dims(p_i, axis=0)
    rho_i = particles.get_densities()
    rho_j = np.expand_dims(rho_i, axis=0)
    v_i = particles.get_velocities()
    v_j = np.expand_dims(v_i, axis=0)
    print("m_i shape:", m_i.shape)
    print("m_j shape:", m_j.shape)
    print("p_i shape:", p_i.shape)
    print("p_j shape:", p_j.shape)
    print("rho_i shape:", rho_i.shape)
    print("rho_j shape:", rho_j.shape)
    print("v_i shape:", v_i.shape)
    print("v_j shape:", v_j.shape)

    m_j_exp = m_j[:, :, np.newaxis]  # (1, N, 1) -> (N, N, 1)
    pressure_term = (p_i / rho_i**2) + (p_j / rho_j**2)
    pressure_term_exp = pressure_term[:, :, np.newaxis]  # (N, N, 1)
    v_i_exp = v_i[np.newaxis, :, :]  # (1, N, 3)
    v_j_exp = v_j[:, :, :]           # (N, N, 3)

    rho_i = np.sum(m_j * w_ij, axis=1)
    # Velocity derivative
    v_deriv = -np.sum(m_j_exp * pressure_term_exp * w_ij_deriv, axis=1)  # (N, 3)
    # Energy derivative
    e_deriv = 0.5 * np.sum(m_j_exp * pressure_term_exp * (v_i_exp - v_j_exp) * w_ij_deriv, axis=1)  # (N, 3)
    # Position derivative
    r_deriv = v_i  # (N, 3)

    return v_deriv, e_deriv, r_deriv


def rk4(f, tspan, y0, h):
    nsteps = int((tspan[1] - tspan[0]) / h)
    N = len(y0)
    t = np.zeros(nsteps + 1)
    y = np.zeros((N, nsteps + 1))

    t[0] = tspan[0]
    y[:,0] = y0
    
    for n in range(nsteps):
        k1 = f(t[n], y[:,n])
        k2 = f(t[n] + 0.5 * h, y[:,n] + 0.5 * h * k1)
        k3 = f(t[n] + 0.5 * h, y[:,n] + 0.5 * h * k2)
        k4 = f(t[n] + h, y[:,n] + h * k3)
        y[:,n+1] = y[:,n] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        t[n+1] = t[n] + h

    return t, y.T



