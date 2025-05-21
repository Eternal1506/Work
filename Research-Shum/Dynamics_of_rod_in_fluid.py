import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters
N = 50
L = 1.0
ds = L / (N - 1)
EI = 1.0
xi_perp = 5.0
xi_par = 2.0
dt = 0.01
T_max = 2.0
num_steps = 500

# Initial configuration: bent and twisted rod
s = np.linspace(0, L, N)
positions = np.zeros((N, 3))
positions[:, 0] = s
positions[:, 1] = 0.2 * np.sin(2 * np.pi * s / L)
positions[:, 2] = 0.1 * np.cos(2 * np.pi * s / L)

def compute_tangents(pos):
    tangents = np.diff(pos, axis=0) / ds
    tangents = np.vstack([tangents, tangents[-1]])  # duplicate last for shape
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents  # normalize for consistent RFT

def compute_elastic_forces(pos):
    tangents = compute_tangents(pos)
    curvature = np.diff(tangents, axis=0) / ds
    curvature = np.vstack([[0, 0, 0], curvature])

    bending_forces = EI * np.diff(np.vstack([curvature, [0, 0, 0]]), axis=0) / ds
    bending_forces = np.vstack([[0,0,0], bending_forces[:-1]])

    return bending_forces

def rod_dynamics(t, y):
    pos = y.reshape((N, 3))
    elastic_forces = compute_elastic_forces(pos)
    tangents = compute_tangents(pos)

    vel = np.zeros_like(pos)
    for i in range(N):
        t_vec = tangents[i]
        f = elastic_forces[i]
        f_par = np.dot(f, t_vec) * t_vec
        f_perp = f - f_par
        vel[i] = f_par / xi_par + f_perp / xi_perp

    return vel.flatten()

# Integrate the system
y0 = positions.flatten()
solution = solve_ivp(rod_dynamics, [0, T_max], y0, method='RK45',
                     max_step=dt, t_eval=np.linspace(0, T_max, num_steps))

# 3D Visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

line, = ax.plot([], [], [], 'o-', lw=2, color='blue')

def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, L)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Kirchhoff Rod Relaxation in Viscous Fluid")
    ax.set_box_aspect([1,1,1])
    return line,

def animate(i):
    pos = solution.y[:, i].reshape((N, 3))
    line.set_data(pos[:, 0], pos[:, 1])
    line.set_3d_properties(pos[:, 2])
    return line,

ani = FuncAnimation(fig, animate, frames=num_steps, init_func=init, blit=False, interval=30)
plt.tight_layout()
plt.show()
