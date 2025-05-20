import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters
L = 2*np.pi
N = 200
s_eval = np.linspace(0, L, N)
ds = L / (N-1)
dt = 0.05
num_steps = 500

# Drag coefficients (anisotropic)
zeta_perp = 5.0
zeta_par = 2.5

# Prescribed curvature (time-dependent actuation)
def curvature(s, t):
    return np.array([
        0.3 * np.sin(0.5 * t + s),
        0.3 * np.cos(0.5 * t + 0.5 * s),
        0.0 
    ])

# Hat operator
def hat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# ODE system in s for rod kinematics (given curvature)
def rod_ode(s, y, t):
    X = y[0:3]
    R = y[3:12].reshape((3,3))
    kappa = curvature(s, t)
    dx_ds = R[:, 2]
    kappa_hat = hat(kappa)
    dR_ds = R @ kappa_hat

    dyds = np.zeros(12)
    dyds[0:3] = dx_ds
    dyds[3:12] = dR_ds.flatten()
    return dyds

# Initial conditions
x0 = np.array([0,0,0])
R0 = np.eye(3)
y0 = np.concatenate((x0, R0.flatten()))

# Initialize rod position history
X_prev = np.zeros((3, N))

# Set up figure
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
arrow_length = 0.3
line, = ax.plot([], [], [], 'b-', linewidth=2)
force_quivers = []

def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-1, L+1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Prescribed Rod Actuation + RFT Forces')
    ax.set_box_aspect([1,1,1])
    return line,

def update(frame):
    global X_prev, force_quivers
    t = frame * dt

    # Clear old force arrows
    for fq in force_quivers:
        fq.remove()
    force_quivers.clear()

    # Integrate rod shape at time t
    def ode_wrapper(s, y): return rod_ode(s, y, t)
    sol = solve_ivp(ode_wrapper, (0, L), y0, t_eval=s_eval)

    X_now = sol.y[0:3, :]

    # Plot rod centerline
    line.set_data(X_now[0,:], X_now[1,:])
    line.set_3d_properties(X_now[2,:])

    # Compute velocities (finite difference)
    V = (X_now - X_prev) / dt

    # Compute RFT drag forces (anisotropic)
    for i in range(0, N, 20):
        R = sol.y[3:12, i].reshape((3,3))
        d3 = R[:,2]
        v_par = np.dot(V[:,i], d3) * d3
        v_perp = V[:,i] - v_par
        F_drag = -zeta_perp * v_perp - zeta_par * v_par

        fq = ax.quiver(X_now[0,i], X_now[1,i], X_now[2,i],
                       F_drag[0], F_drag[1], F_drag[2],
                       length=0.3, color='m')
        force_quivers.append(fq)

    # Update previous positions
    X_prev = X_now.copy()

    return line, *force_quivers

# Animate
anim = FuncAnimation(fig, update, init_func=init, frames=num_steps, interval=50, blit=False)
plt.tight_layout()
plt.show()