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
r_0 = np.concatenate((np.linspace(-0.6, 0, 320), np.linspace(0, 0.6, 80)))
r_0 = np.stack((r_0, np.zeros_like(r_0), np.zeros_like(r_0)), axis=1)  # shape (N, 3)
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


    # print("r:", r.shape)
    # print("r_i:", r_i.shape)
    # print("r_j:", r_j.shape)
    # print("dr:", dr.shape)
    # print("r_abs:", r_abs.shape)
    # print("R:", R.shape)

    w_ij = np.zeros_like(R)
    mask_nearest = (R >= 0) & (R < 1)
    mask_near = (R >= 1) & (R < 2)
    w_ij[mask_nearest] = alpha * ((2/3) - R[mask_nearest]**2 + 0.5 * R[mask_nearest]**3)
    w_ij[mask_near] = alpha * (1/6) * (2 - R[mask_near])**3

    # im0 = axes[0].imshow(mask_nearest, interpolation="nearest", origin="upper")
    # axes[0].set_title("mask_nearest")
    # im1 = axes[1].imshow(mask_near, interpolation="nearest", origin="upper")
    # axes[1].set_title("mask_near")

    R = np.repeat(R[..., np.newaxis], 3, axis=2)
    r_abs = np.repeat(r_abs[..., np.newaxis], 3, axis=2)

    w_ij_deriv = np.zeros_like(R)
    mask_nearest = (R >= 0) & (R < 1)
    mask_near = (R >= 1) & (R < 2)
    w_ij_deriv[mask_nearest] = alpha * (-2 + (1.5)*R[mask_nearest]) * (dr[mask_nearest]/(smoothing_length**2))
    w_ij_deriv[mask_near] = -alpha * 0.5 * (2 - R[mask_near])**2 * (dr[mask_near]/(smoothing_length*r_abs[mask_near]))

    return w_ij, w_ij_deriv



# im1 = axes[2].imshow(w_ij.T, interpolation="nearest", origin="upper")
# axes[2].set_title("Matrix")
# fig.colorbar(im1, ax=axes[2])
# plt.tight_layout()
# plt.show()

# print("w_ij shape:", w_ij.shape)
# print("w_ij_deriv shape:", w_ij_deriv.shape)

eta = 1.3
alpha_n = 1
beta_n = 1

def sph_derivatives(t, y):
    particles.set_from_state_vector(y)
    x, v, e, rho, p = particles.get_from_state_vector(y)
    w_ij, w_ij_deriv = set_w(alpha=1/smoothing_length, x=x)

    # print("x shape:", x.shape)
    # print("v shape:", v.shape)
    # print("e shape:", e.shape)
    # print("rho shape:", rho.shape)
    # print("p shape:", p.shape)

    m = particles.get_mass()
    m_i = np.expand_dims(m, axis=1)
    m_j = np.expand_dims(m, axis=0)

    rho_i = np.sum(m_j * w_ij, axis=1)
    rho = np.squeeze(rho_i)
    p = gammam1 * rho * e
    particles.set_density_pressure(np.squeeze(rho), p)

    particles.save_statevector_to_history(t, np.concatenate([x.flatten(), v.flatten(), e, rho , p]))

    p_i = np.expand_dims(p, axis=1)
    p_j = np.expand_dims(p, axis=0)
    x_i = np.expand_dims(x, axis=1)
    x_j = np.expand_dims(x, axis=0)
    rho_i = np.expand_dims(rho, axis=1)
    rho_j = np.expand_dims(rho, axis=0)
    v_i = np.expand_dims(v, axis=1)
    v_j = np.expand_dims(v, axis=0)
    # print("m_i shape:", m_i.shape)
    # print("m_j shape:", m_j.shape)
    # print("p_i shape:", p_i.shape)
    # print("p_j shape:", p_j.shape)
    # print("rho_i shape:", rho_i.shape)
    # print("rho_j shape:", rho_j.shape)
    # print("v_i shape:", v_i.shape)
    # print("v_j shape:", v_j.shape)
    v_ij = v_i - v_j
    x_ij = x_i - x_j


    # Calculate viscosity
    c = np.sqrt(gammam1*e)
    c_i = np.expand_dims(c, 1)
    c_j = np.expand_dims(c, 0)
    h_ij = smoothing_length
    rho_bar_ij = 0.5*(rho_i + rho_j)
    c_bar_ij = 0.5*(c_i + c_j)

    v_ij_dot_x_ij = np.sum(v_ij*x_ij, axis=2)
    small_phi = 0.1*h_ij
    phi_ij = (h_ij * v_ij_dot_x_ij) / (np.sum(x_ij**2, axis=2) + small_phi**2)
    mask_pi = v_ij_dot_x_ij < 0


    pi_ij = np.zeros((N, N))
    pi_ij[mask_pi] = (
        -alpha_n * c_bar_ij[mask_pi] * phi_ij[mask_pi] + beta_n * phi_ij[mask_pi] ** 2
    ) / rho_bar_ij[mask_pi]


    pp_bracket = ( (p_i/rho_i**2) + (p_j/rho_j**2) + pi_ij)
    left_side = m_j * pp_bracket
    left_side_expanded = np.expand_dims(left_side, axis=2)
    v_deriv = - np.sum(left_side_expanded * w_ij_deriv, axis=1)
    dot = np.sum(v_ij * w_ij_deriv, axis=2)
    e_deriv = 0.5 * np.sum(left_side * dot, axis=1)
    r_deriv = v

    # print("v_deriv: ", v_deriv.shape)
    # print("e_deriv: ", e_deriv.shape)
    # print("r_deriv: ", r_deriv.shape)

    # v_deriv = -np.sum(m_j * ( (p_i/rho_i**2) + (p_j/rho_j**2) + 0 ) * w_ij_deriv, axis=1)
    # e_deriv = 0.5 * np.sum(m_j * ( (p_i/rho_i**2) + (p_j/rho_j**2) + 0 ) * (v_i - v_j) * w_ij_deriv, axis=1)
    # r_deriv = v_i

    return np.concatenate([r_deriv.flatten(), v_deriv.flatten(), e_deriv, np.zeros(N), np.zeros(N)])


t_end = 0.4
n_evals = 800
t_eval = np.linspace(0, t_end, n_evals)
print(t_eval)
sol = integrate.RK45(fun=sph_derivatives, t0=0.0, y0=y0, t_bound=0.4, rtol=1e-6, atol=1e-8, max_step=250)
while sol.status == 'running':
    sol.step()
    if sol.status == 'finished':
        break

# --- Animation of all timestamps ---
import matplotlib.animation as animation

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Energy (e)
line_e, = axs[0, 0].plot([], [], label='Energy (e)')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('e')
axs[0, 0].set_title('Energy')
axs[0, 0].set_xlim(np.min(r_0[:, 0]), np.max(r_0[:, 0]))
axs[0, 0].set_ylim(np.min(e_0), 2.75)
axs[0, 0].legend()

# Velocity (vx)
line_v, = axs[0, 1].plot([], [], label='Velocity (vx)')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('vx')
axs[0, 1].set_title('Velocity')
axs[0, 1].set_xlim(np.min(r_0[:, 0]), np.max(r_0[:, 0]))
axs[0, 1].set_ylim(-0.075, 0.075)
axs[0, 1].legend()

# Density (rho)
line_rho, = axs[1, 0].plot([], [], label='Density (rho)')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('rho')
axs[1, 0].set_title('Density')
axs[1, 0].set_xlim(np.min(r_0[:, 0]), np.max(r_0[:, 0]))
axs[1, 0].set_ylim(0, 1.25)
axs[1, 0].legend()

# Pressure (p)
line_p, = axs[1, 1].plot([], [], label='Pressure (p)')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('p')
axs[1, 1].set_title('Pressure')
axs[1, 1].set_xlim(np.min(r_0[:, 0]), np.max(r_0[:, 0]))
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
ani.save('sodshock_movie.mp4', writer='ffmpeg')

plt.show()
