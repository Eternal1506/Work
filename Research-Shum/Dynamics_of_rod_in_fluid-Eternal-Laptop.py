import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters
N = 100
L = 10.0
ds = L / (N-1)
timesteps = 500
dt = 0.004
zeta_perp = 2.0
zeta_para = 1.0
bending_modulus = 3.0
twist_modulus = 1.0

# Initialize positions
X = np.zeros((timesteps, N, 3))

# Sinusoidal initial condition
# X[0,:,2] = np.linspace(0, L, N)
# X[0,:,0] = 1 * np.sin(2*np.pi*np.linspace(0, L, N)/L)

# Circular initial condition
R = L / (2 * np.pi)  
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
X[0,:,0] = R * np.cos(theta)
X[0,:,1] = R * np.sin(theta)
X[0,:,2] = 0

# Straight initial condition
# X[0,:,0] = np.linspace(-L/2, L/2, N)
# X[0,:,1] = np.linspace(-L/2, L/2, N)
# X[0,:,2] = 2

# Small perturbation
# xi = 0.5
# X[0,:,0] = 0
# X[0,:,1] = 0
# X[0,:,2] = (1 + xi) * np.linspace(0, L, N)



# Initialize directors
D1 = np.zeros((timesteps, N, 3))
D2 = np.zeros((timesteps, N, 3))
D3 = np.zeros((timesteps, N, 3))

# Initial directors for straight rod and sinusoidal rod
D1[0,:,0] = np.cos(theta)
D1[0,:,1] = np.sin(theta)
D2[0,:,2] = -1
D3[0,:,0] = -np.sin(theta)
D3[0,:,1] = np.cos(theta)

# Initial directors for small perturbation
# D1[0,:,0] = 1.0
# D2[0,:,1] = np.cos(xi)
# D2[0,:,2] = -np.sin(xi)
# D3[0,:,1] = np.sin(xi)
# D3[0,:,2] = np.cos(xi)

# Compute bending forces
def compute_forces(X_t):
    F = np.zeros_like(X_t)
    for i in range(1, N-1):
        d2X_ds2 = (X_t[i+1] -  X_t[i-1]) / (2*ds)
        F[i] = bending_modulus * d2X_ds2
    return F

# Compute torques (simplified)
def compute_torques(D3_t):
    T = np.zeros_like(D3_t)
    for i in range(1, N-1):
        dD3_ds = (D3_t[i+1] - D3_t[i-1]) / (2*ds)
        T[i] = twist_modulus * np.cross(D3_t[i], dD3_ds)
    return T

# Time stepping
for t in range(timesteps-1):
    F = compute_forces(X[t])
    Tq = compute_torques(D3[t])

    V = np.zeros_like(X[t])
    W = np.zeros_like(X[t])

    # print("Force:", F)
    # print("Torque:", Tq)

    for i in range(1, N-1):
        # Local tangent
        t_hat = (X[t,i+1] - X[t,i-1]) / (2*ds)
        t_hat /= np.linalg.norm(t_hat)
        I = np.eye(3)

        # Local mobility tensor from RFT
        M = (1/zeta_para) * np.outer(t_hat, t_hat) + (1/zeta_perp) * (I - np.outer(t_hat, t_hat))

        # Velocity from force
        V[i] = M @ F[i]

        # Angular velocity from torque (simplified: W = T / rotational drag)
        W[i] = 0.5 * Tq[i]  # arbitrary scaling

    # Update positions
    X[t+1] = X[t] + V * dt

    # Update directors
    for i in range(N):
        omega = W[i]
        for D in [D1, D2, D3]:
            D[t+1,i] = D[t,i] + np.cross(omega, D[t,i]) * dt
            D[t+1,i] /= np.linalg.norm(D[t+1,i])

# Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], 'b-', linewidth=2)

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

def update(frame):
    data = X[frame]
    line.set_data(data[:,0], data[:,1])
    line.set_3d_properties(data[:,2])
    return line,

ani = FuncAnimation(fig, update, frames=timesteps, init_func=init, blit=True, interval=30)
plt.show()
