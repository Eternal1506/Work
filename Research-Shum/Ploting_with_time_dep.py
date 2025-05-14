import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -----------------------------
# Define the time-dependent curvature function
# -----------------------------
def curvature(s, t):
    # Example: oscillating curvature around y-axis over time
    return np.array([
        0.3 * np.sin(0.5 * t + s),
        0.3 * np.cos(0.5 * t + 0.5 * s),
        0.0
    ])

# -----------------------------
# ODE system for rod kinematics
# -----------------------------
def rod_ode(s, y, t):
    x = y[:3]                    # Position vector
    R = y[3:].reshape((3,3))     # Rotation matrix
    kappa = curvature(s,t)

    # Derivative of position
    dx_ds = R[:,2]  # d3 is the third director (tangent)

    # Skew-symmetric matrix of curvature
    kappa_hat = np.array([
        [0,        -kappa[2],  kappa[1]],
        [kappa[2],  0,        -kappa[0]],
        [-kappa[1], kappa[0],  0       ]
    ])

    # Derivative of rotation matrix
    dR_ds = R @ kappa_hat

    # Flatten and concatenate derivatives
    dyds = np.concatenate((dx_ds, dR_ds.flatten()))
    return dyds

# -----------------------------
# Initial Conditions
# -----------------------------
x0 = np.array([0, 0, 0])
R0 = np.eye(3)
y0 = np.concatenate((x0, R0.flatten()))

# -----------------------------
# Integration parameters
# -----------------------------
s_span = (0, 2*np.pi)
s_eval = np.linspace(s_span[0], s_span[1], 500)

# -----------------------------
# Animation setup
# -----------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
arrow_length = 0.2

line, = ax.plot([], [], [], 'b', linewidth=2)
quivers = []

def init():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Time-dependent Rod Simulation')
    ax.set_box_aspect([1,1,1])
    return line,

def update(frame):
    global quivers
    t = frame * 0.1

    # Clear previous arrows
    for q in quivers:
        q.remove()
    quivers.clear()

    # Integrate for current time t
    def ode_wrapper(s, y): return rod_ode(s, y, t)
    sol = solve_ivp(ode_wrapper, s_span, y0, t_eval=s_eval)

    X = sol.y[0, :]
    Y = sol.y[1, :]
    Z = sol.y[2, :]

    line.set_data(X, Y)
    line.set_3d_properties(Z)

    # Plot directors at selected points
    for i in range(0, len(s_eval), 50):
        R = sol.y[3:, i].reshape((3,3))
        pos = [X[i], Y[i], Z[i]]
        for j in range(3):
            q = ax.quiver(pos[0], pos[1], pos[2],
                          R[0,j], R[1,j], R[2,j],
                          length=arrow_length, color=['r','g','k'][j])
            quivers.append(q)

    return line, *quivers

# -----------------------------
# Run the animation
# -----------------------------
anim = FuncAnimation(fig, update, init_func=init, frames=200, interval=50, blit=False)

plt.tight_layout()
plt.show()
