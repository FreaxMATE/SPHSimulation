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

gammam1 = 1.4 - 1
smoothing_length = 1e7


# Load initial conditions

data = np.loadtxt('/home/kunruh/Documents/Studium/Physik/Master/4/AppliedCompPhysicsAndML/Project/SPHSimulation/2/Planet2400.dat')
dx0_planet1 = 2.5e8
dv0_planet1 = 4e6
data[:, :3] -= dx0_planet1
data[:, 3:6] += dv0_planet1

dx0_planet2 = 2.5e8
dv0_planet2 = 4e6
data_1 = data.copy()
data_1[:, :3] += dx0_planet2
data_1[:, 3:6] -= dv0_planet2

data = np.concatenate([data, data_1])

N = len(data)
x_0 = np.array([data[:, 0], data[:, 1], data[:, 2]]).T
v_0 = np.array([data[:, 3], data[:, 4], data[:, 5]]).T
m_0 = np.array(data[:, 6])
rho_0 = np.array(data[:, 7])
p_0 = np.array(data[:, 8])
e_0 = p_0 / (rho_0 * gammam1)

particles = Particles(x=x_0, v=v_0, e=e_0, rho=rho_0, p=p_0, m=m_0, N=N)
y0 = particles.to_state_vector()

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


    mask_nearest = (R >= 0) & (R <= 1)
    mask_near = (R >= 1) & (R <= 2)
    mask_far = R >= 2

    phi_deriv = np.zeros_like(R)
    phi_deriv[mask_nearest] = (1/smoothing_length**2)*((4/3)*R[mask_nearest] - (6/5)*R[mask_nearest]**3 + 0.5*R[mask_nearest]**4)
    phi_deriv[mask_near] = (1/smoothing_length**2)*((8/3)*R[mask_near] - 3*R[mask_near]**2 + (6/5)*R[mask_near]**3 - (1/6)*R[mask_near]**4 - 1/(15*R[mask_near]**2))
    phi_deriv[mask_far] = 1/r_abs[mask_far]**2

    R = np.repeat(R[..., np.newaxis], 3, axis=2)
    r_abs_exp = np.repeat(r_abs[..., np.newaxis], 3, axis=2)

    w_ij_deriv = np.zeros_like(R)
    mask_nearest = (R >= 0) & (R < 1)
    mask_near = (R >= 1) & (R < 2)
    w_ij_deriv[mask_nearest] = alpha * (-2 + (1.5)*R[mask_nearest]) * (dr[mask_nearest]/(smoothing_length**2))
    w_ij_deriv[mask_near] = -alpha * 0.5 * (2 - R[mask_near])**2 * (dr[mask_near]/(smoothing_length*r_abs_exp[mask_near]))

    return w_ij, w_ij_deriv, phi_deriv, r_abs

eta = 1.3
alpha_n = 1
beta_n = 1

def sph_derivatives(t, y):
    # Set state from vector
    particles.set_from_state_vector(y)
    x, v, e, _, _ = particles.get_from_state_vector(y)
    m = particles.get_mass()
    # w_ij, w_ij_deriv = set_w(alpha=3/(2*np.pi*smoothing_length**3), x=x)
    w_ij, w_ij_deriv, phi_deriv, r_abs = set_w(alpha=3/(2*np.pi*smoothing_length**3), x=x)

    # Compute density and pressure
    rho = np.sum(m[None, :] * w_ij, axis=1)
    p = gammam1 * rho * e
    particles.set_density_pressure(rho, p)

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

    # Gravitational acceleration - exclude self-interaction (i==j)
    mask_not_self = ~np.eye(N, dtype=bool)
    grav_force_term = np.zeros((N, N, 3))
    grav_force_term[mask_not_self] = (m[None, :, None] * phi_deriv[:, :, None] * x_ij / r_abs[:, :, None])[mask_not_self]
    v_grav_deriv = -constants.gravitational_constant * np.sum(grav_force_term, axis=1)

    # Velocity derivative
    v_deriv = -np.einsum('ij,ijk->ik', left_side, w_ij_deriv) + v_grav_deriv

    # Energy derivative
    dot = np.einsum('ijk,ijk->ij', v_ij, w_ij_deriv)
    e_deriv = 0.5 * np.sum(left_side * dot, axis=1)

    # Position derivative
    r_deriv = v
    if int(t) % 100 == 0:
        print(f"Progress: t = {t:.2f}")
    return np.concatenate([r_deriv.flatten(), v_deriv.flatten(), e_deriv, np.zeros(N), np.zeros(N)])


sol = integrate.RK45(fun=sph_derivatives, t0=0.0, y0=y0, t_bound=800, rtol=1e-6, atol=1e-8)

# Save initial state
particles.save_statevector_to_history(sol.t, sol.y)

while sol.status == 'running':
    sol.step()
    # Save state at each completed step
    if sol.status == 'running' or sol.status == 'finished':
        # Get current state and compute density/pressure for history
        particles.set_from_state_vector(sol.y)
        x, v, e, _, _ = particles.get_from_state_vector(sol.y)
        m = particles.get_mass()
        w_ij, _, _, _ = set_w(alpha=3/(2*np.pi*smoothing_length**3), x=x)
        rho = np.sum(m[None, :] * w_ij, axis=1)
        p = gammam1 * rho * e
        
        particles.save_statevector_to_history(sol.t, np.concatenate([x.flatten(), v.flatten(), e, rho, p]))
    
    if sol.status == 'finished':
        break

# --- Animation of all timestamps ---
import matplotlib.animation as animation

fig, axs = plt.subplots(figsize=(12, 8))

# Set up the plot
axs.set_xlabel('Position x / m')
axs.set_ylabel('Position y / m')

# Get initial data to set up plot limits
x_init, v_init, e_init, rho_init, p_init = particles.get_from_state_vector(particles.sv_history[0][1])

# Calculate plot limits based on middle frame data
middle_frame_index = len(particles.sv_history) // 2
t_middle, state_vector_middle = particles.sv_history[middle_frame_index]
x_middle, v_middle, e_middle, rho_middle, p_middle = particles.get_from_state_vector(state_vector_middle)

x_min, x_max = 2*np.min(x_middle[:, 0]), 2*np.max(x_middle[:, 0])
y_min, y_max = 2*np.min(x_middle[:, 1]), 2*np.max(x_middle[:, 1])

# Still calculate rho limits from all data for consistent coloring
all_rho_values = []
for t, state_vector in particles.sv_history:
    x, v, e, rho, p = particles.get_from_state_vector(state_vector)
    all_rho_values.extend(rho)

rho_min, rho_max = min(all_rho_values), max(all_rho_values)

# Add some padding
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

axs.set_xlim(x_min - x_padding, x_max + x_padding)
axs.set_ylim(y_min - y_padding, y_max + y_padding)

# Initial scatter plot
scat = axs.scatter(x_init[:, 0], x_init[:, 1], c=rho_init, cmap='viridis', vmin=rho_min, vmax=rho_max, s=20)

# Add colorbar
cbar = plt.colorbar(scat, ax=axs)
cbar.set_label(r'Density $\rho$ / $kg/m^{3}$')

# Title
title = axs.set_title(f"SPH Simulation - Density at t={particles.sv_history[0][0]:.4f} s")

def animate(frame):
    """Animation function called for each frame"""
    if frame < len(particles.sv_history):
        t, state_vector = particles.sv_history[frame]
        x, v, e, rho, p = particles.get_from_state_vector(state_vector)
        
        # Update scatter plot data
        scat.set_offsets(np.column_stack((x[:, 0], x[:, 1])))
        scat.set_array(rho)
        
        # Update title
        title.set_text(f"SPH Simulation - Density at t={t:.4f} s")
    
    return scat, title

# Create animation
print(f"Creating animation with {len(particles.sv_history)} frames...")
anim = animation.FuncAnimation(fig, animate, frames=len(particles.sv_history), interval=50, blit=False, repeat=True)

# Save as MP4

filename = 'planets_2400'
filename += f"_dx1_{int(dx0_planet1)}_dv1_{int(dv0_planet1)}_dx2_{int(dx0_planet2)}_dv2_{int(dv0_planet2)}"
print(f"Saving animation as {filename}...")

anim.save(filename+'.mp4', writer='ffmpeg', fps=20, bitrate=1800)
np.savetxt(filename+'.csv', y0)
plt.tight_layout()
plt.show()

print("Animation saved successfully!")
