import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Define the curvature function
# -----------------------------
def curvature(s):
    # Constant curvature: bends around y-axis with kappa2 = 1
    return np.array([0.2*np.sin(s), 0.2*np.cos(s), 0.2*s^3])  # Example: circular arc in x-y plane

# -----------------------------
# ODE system for rod kinematics
# -----------------------------
def rod_ode(s, y):
    x = y[:3]                    # Position vector
    R = y[3:].reshape((3,3))     # Rotation matrix
    kappa = curvature(s)

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
x0 = np.array([0, 0, 0])         # Starting position
R0 = np.eye(3)                   # Starting orientation (identity)
y0 = np.concatenate((x0, R0.flatten()))

# -----------------------------
# Integration parameters
# -----------------------------
# Integrating over 2π for a perfect circle when kappa = 1
s_span = (0, 2*np.pi)
s_eval = np.linspace(s_span[0], s_span[1], 500)

# -----------------------------
# Solve the ODE
# -----------------------------
sol = solve_ivp(rod_ode, s_span, y0, t_eval=s_eval)

# -----------------------------
# Extract solution
# -----------------------------
X = sol.y[0, :]
Y = sol.y[1, :]
Z = sol.y[2, :]

# -----------------------------
# Plot curve and directors
# -----------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Plot centerline
ax.plot(X, Y, Z, color='b', linewidth=2, label='Rod Centerline')

show_arrows = True  # Set to False to hide arrows
if show_arrows:
    # Plot directors (arrows) along the rod
    arrow_length = 0.5  # Length of arrows
    for i in range(0, len(s_eval), 30):  # Adjust step for fewer arrows
        R = sol.y[3:, i].reshape((3,3))
        pos = [X[i], Y[i], Z[i]]
        for j in range(3):
            ax.quiver(pos[0], pos[1], pos[2],
                      R[0,j], R[1,j], R[2,j], length=arrow_length, normalize=True, color=['r','g','k'][j])
# # Plot directors (arrows) at selected points
# arrow_length = 0.5  # much shorter, looks better relative to rod size
# for i in range(0, len(s_eval), 30):
#     R = sol.y[3:, i].reshape((3,3))
#     pos = [X[i], Y[i], Z[i]]
#     for j in range(3):
#         ax.quiver(pos[0], pos[1], pos[2],
#                   R[0,j], R[1,j], R[2,j], length = arrow_length, normalize = True, color=['r','g','k'][j])

# -----------------------------
# Plot settings
# -----------------------------
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Rod Simulation')
ax.legend()
ax.grid(True)
ax.set_box_aspect([1,1,1])  # Equal aspect ratio
plt.tight_layout()
plt.show()
