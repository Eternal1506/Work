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

# Integrate rod shape in s at fixed time t, given initial conditions y0
def integrate_rod(t, y0):
    def rod_ode(s, y):
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
    sol = solve_ivp(rod_ode, (0, L), y0, t_eval=s_eval, vectorized=False)
    Xs = sol.y[0:3, :]
    Rs = np.zeros((3,3,N))
    for i in range(N):
        Rs[:,:,i] = sol.y[3:12, i].reshape((3,3))
    return Xs, Rs

# Compute drag forces per segment from velocities and rotations
def compute_drag_forces(V, Rs):
    F_drag = np.zeros_like(V)
    for i in range(N):
        d3 = Rs[:,2,i]
        v_par = np.dot(V[:,i], d3) * d3
        v_perp = V[:,i] - v_par
        F_drag[:,i] = -zeta_perp * v_perp - zeta_par * v_par
    return F_drag

# Initialize rod state: position and rotation at s=0
x0 = np.array([0,0,0])
R0 = np.eye(3)
y0 = np.concatenate((x0, R0.flatten()))

# Initialize rod position and rotation along s at time 0
X_now, R_now = integrate_rod(0, y0)
X_prev = X_now.copy()

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
    ax.set_title('Kirchhoff Rod + RFT Drag Forces')
    ax.set_box_aspect([1,1,1])
    return line,

def update(frame):
    global X_now, R_now, X_prev, y0, force_quivers
    t = frame * dt

    # Clear old force arrows
    for fq in force_quivers:
        fq.remove()
    force_quivers.clear()

    # Compute velocity from previous step
    V = (X_now - X_prev) / dt

    # Compute drag forces
    F_drag = compute_drag_forces(V, R_now)

    # For demonstration, we simply assume velocity is proportional to drag force:
    # This is a very simplified force balance model: V = mobility * F_drag
    # In reality, you would solve a more complex elastodynamics + drag system
    mobility = 0.1
    V_new = mobility * F_drag

    # Update rod positions using velocity from force balance
    X_next = X_now + V_new * dt

    # Reconstruct y0 for next integration step from new base position and orientation
    # For simplicity, keep R0 fixed here (no twist update)
    y0 = np.concatenate((X_next[:,0], R0.flatten()))

    # Integrate rod shape with updated base position and time
    X_new, R_new = integrate_rod(t, y0)

    # Update previous and current states
    X_prev = X_now
    X_now = X_new
    R_now = R_new

    # Plot rod centerline
    line.set_data(X_now[0,:], X_now[1,:])
    line.set_3d_properties(X_now[2,:])

    # Plot drag force arrows every 20 points
    for i in range(0, N, 20):
        fq = ax.quiver(X_now[0,i], X_now[1,i], X_now[2,i],
                       F_drag[0,i], F_drag[1,i], F_drag[2,i],
                       length=arrow_length, color='m')
        force_quivers.append(fq)

    return line, *force_quivers

anim = FuncAnimation(fig, update, init_func=init, frames=num_steps, interval=50, blit=False)
plt.tight_layout()
plt.show()
