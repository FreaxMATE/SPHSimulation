import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

class Particles():
    def __init__(self, x, v, e, rho, p, m, N):
        self.x = x
        self.v = v
        self.e = e
        self.rho = rho
        self.p = p
        self.m = m
        self.N = N
        self.sv_history = [] # state vector history

    def to_state_vector(self):
        return np.concatenate([self.x.flatten(), self.v.flatten(), self.e, self.rho, self.p])

    def set_from_state_vector(self, state_vector):
        self.x = np.reshape(state_vector[:3*self.N], (self.N, 3))
        self.v = np.reshape(state_vector[3*self.N:6*self.N], (self.N, 3))
        self.e = state_vector[6*self.N:7*self.N]
        self.rho = state_vector[7*self.N:8*self.N]
        self.p = state_vector[8*self.N:9*self.N]

    def set_density_pressure(self, rho, p):
        self.rho = rho
        self.p = p

    def get_from_state_vector(self, state_vector):
        # x, v, e, rho, p
        x = np.reshape(state_vector[:3*self.N], (self.N, 3))
        v = np.reshape(state_vector[3*self.N:6*self.N], (self.N, 3))
        e = state_vector[6*self.N:7*self.N]
        rho = state_vector[7*self.N:8*self.N]
        p = state_vector[8*self.N:9*self.N]
        return x, v, e, rho, p

    def get_mass(self):
        return self.m

    def get(self):
        return self.x, self.v, self.e, self.rho, self.p

    def save_statevector_to_history(self, t, state_vector):
        self.sv_history.append((t, state_vector))

    def get_statevector_history(self):
        return self.sv_history

# Coordinates of particles
N = 400
r_0 = np.concatenate((np.linspace(-0.6, 0, 320), np.linspace(0, 0.6, 81)[1:]))
r_0 = np.stack((r_0, np.zeros_like(r_0), np.zeros_like(r_0)), axis=1)
for i in r_0:
    print(i)
v_0 = np.zeros((N, 3))
e_0 = np.concatenate((np.full(320, 2.5), np.full(N - 320, 1.795)))
rho_0 = np.concatenate((np.full(320, 1.0), np.full(N - 320, 0.25)))
p_0 = np.concatenate((np.full(320, 1.0), np.full(N - 320, 0.1795)))
m_0 = np.full(N, 0.001875)
particles = Particles(x=r_0, v=v_0, e=e_0, rho=rho_0, p=p_0, m=m_0, N=N)
y0 = particles.to_state_vector()
gammam1 = 1.4 - 1
smoothing_length = 0.01


# fig, axes = plt.subplots(1, 3, figsize=(10, 4))

def set_w(alpha, x):
    r_i = np.expand_dims(x, axis=1)
    r_j = np.expand_dims(x, axis=0)

    # Distance between particles in x, y, z direction
    dr = r_i - r_j
    # Absolute 3D distance between particles
    r_abs = np.linalg.norm(dr, axis=2)
    R = r_abs / smoothing_length
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
    w_ij_deriv[mask_nearest] = alpha * (-2 + (1.5)*R[mask_nearest]) * (dr[mask_nearest]/(smoothing_length**2))
    w_ij_deriv[mask_near] = -alpha * 0.5 * (2 - R[mask_near])**2 * (dr[mask_near]/(smoothing_length*r_abs[mask_near]))

    return w_ij, w_ij_deriv

eta = 1.3
alpha_n = 1
beta_n = 1

def sph_derivatives(t, y):
    # Set state from vector
    particles.set_from_state_vector(y)
    x, v, e, _, _ = particles.get_from_state_vector(y)
    m = particles.get_mass()
    w_ij, w_ij_deriv = set_w(alpha=1/smoothing_length, x=x)

    # Compute density and pressure
    rho = np.sum(m[None, :] * w_ij, axis=1)
    p = gammam1 * rho * e
    particles.set_density_pressure(rho, p)

    # Save state history (optional: comment out to save memory)
    particles.save_statevector_to_history(t, np.concatenate([x.flatten(), v.flatten(), e, rho, p]))

    # Precompute expanded arrays for broadcasting
    p_i = p[:, None]
    p_j = p[None, :]
    rho_i = rho[:, None]
    rho_j = rho[None, :]
    v_i = v[:, None, :]
    v_j = v[None, :, :]
    v_ij = v_i - v_j
    x_i = x[:, None, :]
    x_j = x[None, :, :]
    x_ij = x_i - x_j

    # Viscosity
    c = np.sqrt(gammam1 * e)
    c_i = c[:, None]
    c_j = c[None, :]
    h_ij = smoothing_length
    rho_bar_ij = 0.5 * (rho_i + rho_j)
    c_bar_ij = 0.5 * (c_i + c_j)

    v_ij_dot_x_ij = np.einsum('ijk,ijk->ij', v_ij, x_ij)
    x_ij_norm2 = np.einsum('ijk,ijk->ij', x_ij, x_ij)
    small_phi2 = (0.1 * h_ij) ** 2
    phi_ij = (h_ij * v_ij_dot_x_ij) / (x_ij_norm2 + small_phi2)
    mask_pi = v_ij_dot_x_ij < 0

    pi_ij = np.zeros((N, N), dtype=np.float64)
    pi_ij[mask_pi] = (
        -alpha_n * c_bar_ij[mask_pi] * phi_ij[mask_pi] + beta_n * phi_ij[mask_pi] ** 2
    ) / rho_bar_ij[mask_pi]

    # Pressure bracket
    pp_bracket = (p_i / (rho_i ** 2)) + (p_j / (rho_j ** 2)) + pi_ij
    left_side = m[None, :] * pp_bracket

    # Velocity derivative
    v_deriv = -np.einsum('ij,ijk->ik', left_side, w_ij_deriv)

    # Energy derivative
    dot = np.einsum('ijk,ijk->ij', v_ij, w_ij_deriv)
    e_deriv = 0.5 * np.sum(left_side * dot, axis=1)

    # Position derivative
    r_deriv = v

    # Return concatenated derivatives
    return np.concatenate([r_deriv.flatten(), v_deriv.flatten(), e_deriv, np.zeros(N), np.zeros(N)])


sol = integrate.RK45(fun=sph_derivatives, t0=0.0, y0=y0, t_bound=0.4, rtol=1e-6, atol=1e-8)
while sol.status == 'running':
    sol.step()
    if sol.status == 'finished':
        break

# --- Animation of all timestamps ---
import matplotlib.animation as animation

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Energy (e)
line_e, = axs[0, 0].plot([], [], label='Energy (e)')
axs[0, 0].set_xlabel('Position x / m')
axs[0, 0].set_ylabel('Internal Energy e / J/kg')
axs[0, 0].set_title('Energy')
axs[0, 0].set_xlim(-0.45, 0.45)
axs[0, 0].set_ylim(np.min(e_0), 2.75)
axs[0, 0].legend()

# Velocity (vx)
line_v, = axs[0, 1].plot([], [], label='Velocity (v)')
axs[0, 1].set_xlabel('Position x / m')
axs[0, 1].set_ylabel('Velocity v / m/s')
axs[0, 1].set_title('Velocity')
axs[0, 1].set_xlim(-0.45, 0.45)
axs[0, 1].set_ylim(-0.075, 1)
axs[0, 1].legend()

# Density (rho)
line_rho, = axs[1, 0].plot([], [], label='Density (rho)')
axs[1, 0].set_xlabel('Position x / m')
axs[1, 0].set_ylabel(r'Density $\rho$ / $kg/m^{3}$')
axs[1, 0].set_title('Density')
axs[1, 0].set_xlim(-0.45, 0.45)
axs[1, 0].set_ylim(0, 1.25)
axs[1, 0].legend()

# Pressure (p)
line_p, = axs[1, 1].plot([], [], label='Pressure (p)')
axs[1, 1].set_xlabel('Position x / m')
axs[1, 1].set_ylabel(r'Pressure / $N/m^2$')
axs[1, 1].set_title('Pressure')
axs[1, 1].set_xlim(-0.45, 0.45)
axs[1, 1].set_ylim(0, 1.25)
axs[1, 1].legend()

plt.tight_layout()

def init():
    line_e.set_data([], [])
    line_v.set_data([], [])
    line_rho.set_data([], [])
    line_p.set_data([], [])
    return line_e, line_v, line_rho, line_p

def animate(i):
    x, v, e, rho, p = particles.get_from_state_vector(particles.sv_history[i][1])
    line_e.set_data(x[:, 0], e)
    line_v.set_data(x[:, 0], v[:, 0])
    line_rho.set_data(x[:, 0], rho)
    line_p.set_data(x[:, 0], p)
    axs[0, 0].set_title(f"Energy at t={particles.sv_history[i][0]:.4f}")
    axs[0, 1].set_title(f"Velocity at t={particles.sv_history[i][0]:.4f}")
    axs[1, 0].set_title(f"Density at t={particles.sv_history[i][0]:.4f}")
    axs[1, 1].set_title(f"Pressure at t={particles.sv_history[i][0]:.4f}")
    return line_e, line_v, line_rho, line_p

ani = animation.FuncAnimation(fig, animate, frames=len(particles.sv_history), init_func=init, blit=False)

# To display the animation in a Jupyter notebook, use:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To save the animation as an mp4 file:

plt.show()

ani.save('sodshock_movie.mp4', writer='ffmpeg')
