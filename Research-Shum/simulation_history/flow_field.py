import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numba

# --- Regularized Stokes kernel scalar functions (from user code) ---
@numba.njit(cache=True)
def H1_func(r, epsilon_reg):
    """Scalar part of the Stokeslet mobility tensor (identity part)."""
    return (2.0 * epsilon_reg ** 2 + r ** 2) / (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (3.0 / 2.0))

@numba.njit(cache=True)
def H2_func(r, epsilon_reg):
    """Scalar part of the Stokeslet mobility tensor (dyadic part)."""
    return 1.0 / (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (3.0 / 2.0))

@numba.njit(cache=True)
def Q_func(r, epsilon_reg):
    """Scalar part of the Rotlet mobility tensor."""
    return (5.0 * epsilon_reg ** 2 + 2.0 * r ** 2) / (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (5.0 / 2.0))

@numba.njit(cache=True)
def D1_func(r, epsilon_reg):
    """Scalar part of the Potential Dipole mobility tensor (identity part)."""
    return (10.0 * epsilon_reg ** 4 - 7.0 * epsilon_reg ** 2 * r ** 2 - 2.0 * r ** 4) / \
           (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (7.0 / 2.0))

@numba.njit(cache=True)
def D2_func(r, epsilon_reg):
    """Scalar part of the Potential Dipole mobility tensor (dyadic part)."""
    return (21.0 * epsilon_reg ** 2 + 6.0 * r ** 2) / \
           (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (7.0 / 2.0))


def hydro_velocity_field(grid_x, grid_y, force_positions, forces, torque_positions, torques, epsilon_vec):
    """
    Calculates fluid velocity on a grid using regularized Stokeslets and Rotlets.

    Args:
        grid_x (np.ndarray): 2D array of X coordinates of the grid points.
        grid_y (np.ndarray): 2D array of Y coordinates of the grid points.
        force_positions (np.ndarray): Positions of forces.
        forces (np.ndarray): Force vectors.
        torque_positions (np.ndarray): Positions of torques.
        torques (np.ndarray): Torque vectors.
        epsilon_vec (np.ndarray): Vector of regularization parameters for each source.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: U, V, velocity magnitude on the grid.
    """
    U = np.zeros_like(grid_x, dtype=float)
    V = np.zeros_like(grid_y, dtype=float)
    grid_z = np.zeros_like(grid_x)

    # 1. Contribution from Forces (Stokeslets)
    for i, (f_pos, f_vec) in enumerate(zip(force_positions, forces)):
        rx, ry, rz = grid_x - f_pos[0], grid_y - f_pos[1], grid_z - f_pos[2]
        r_sq = rx**2 + ry**2 + rz**2
        r = np.sqrt(r_sq)
        eps = epsilon_vec[i]
        
        h1 = H1_func(r, eps)
        h2 = H2_func(r, eps)
        f_dot_r = f_vec[0] * rx + f_vec[1] * ry + f_vec[2] * rz
        
        U += f_vec[0] * h1 + f_dot_r * rx * h2
        V += f_vec[1] * h1 + f_dot_r * ry * h2

    # 2. Contribution from Torques (Rotlets)
    for i, (m_pos, m_vec) in enumerate(zip(torque_positions, torques)):
        rx, ry, rz = grid_x - m_pos[0], grid_y - m_pos[1], grid_z - m_pos[2]
        r_sq = rx**2 + ry**2 + rz**2
        r = np.sqrt(r_sq)
        eps = epsilon_vec[i] # Assumes forces and torques share positions and epsilons

        q_val = Q_func(r, eps)
        
        # cross_mr = m x r
        cross_mr0 = m_vec[1] * rz - m_vec[2] * ry
        cross_mr1 = m_vec[2] * rx - m_vec[0] * rz
        
        U += 0.5 * cross_mr0 * q_val
        V += 0.5 * cross_mr1 * q_val

    magnitude = np.sqrt(U**2 + V**2)
    return U, V, magnitude


def create_flow_animation(history_dir="simulation_history"):
    """
    Loads swimmer simulation data, calculates the surrounding fluid flow, and creates an animation.
    """
    print(f"Looking for simulation data in: '{history_dir}'")

    # --- 1. Define paths and check for files ---
    paths = {
        "params": os.path.join(history_dir, "simulation_params.txt"),
        "X": os.path.join(history_dir, "history_X.txt"),
        "f": os.path.join(history_dir, "history_f.txt"),
        "n": os.path.join(history_dir, "history_n.txt"), # Torques on filament
        "head_X": os.path.join(history_dir, "history_head_X.txt"),
        "head_u": os.path.join(history_dir, "history_head_u.txt"),
        "head_w": os.path.join(history_dir, "history_head_w.txt"), # Head angular velocity
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Error: Required file not found: '{path}'")
            return

    # --- 2. Load Simulation Parameters ---
    try:
        params = {}
        with open(paths["params"], 'r') as f:
            for line in f:
                key, value = line.strip().split(' = ')
                try:
                    if '.' in value or 'e' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
        
        num_frames = int(params['num_frames'])
        M = int(params['M'])
        dt_snapshot = float(params['dt_snapshot'])
        L_eff = params.get('L_eff', 1.0)
        print("Successfully loaded parameters.")
    except Exception as e:
        print(f"Error reading or parsing parameters: {e}")
        return

    # --- 3. Load History Data ---
    try:
        print("Loading simulation history data...")
        history_X = np.loadtxt(paths["X"], skiprows=1).reshape(num_frames, M, 3)
        history_f = np.loadtxt(paths["f"], skiprows=1).reshape(num_frames, M, 3)
        history_n = np.loadtxt(paths["n"], skiprows=1).reshape(num_frames, M, 3)
        history_head_X = np.loadtxt(paths["head_X"], skiprows=1).reshape(num_frames, 3)
        history_head_u = np.loadtxt(paths["head_u"], skiprows=1).reshape(num_frames, 3)
        history_head_w = np.loadtxt(paths["head_w"], skiprows=1).reshape(num_frames, 3)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading history files: {e}")
        return
        
    # --- 4. Calculate Head Forces and Torques on Fluid ---
    segment_length = L_eff / (M - 1)
    R_head = segment_length * 1.5
    
    # Force on fluid = - (translational drag on head)
    # Translational drag F = 6*pi*mu*R*U. Assuming mu=1.
    trans_drag_coeff = 6 * np.pi * R_head
    history_head_f = -trans_drag_coeff * history_head_u

    # Torque on fluid = - (rotational drag on head)
    # Rotational drag T = 8*pi*mu*R^3*w. Assuming mu=1.
    rot_drag_coeff = 8 * np.pi * R_head**3
    history_head_n = -rot_drag_coeff * history_head_w

    # --- 5. Set up the Plot ---
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # --- 6. Define the Animation Update Function ---
    def update_animation(frame):
        ax.clear()

        # Get data for the current frame
        X_t = history_X[frame]
        f_t = history_f[frame]
        n_t = history_n[frame]
        head_X_t = history_head_X[frame]
        head_f_t = history_head_f[frame]
        head_n_t = history_head_n[frame]
        
        # Combine filament and head data for calculation
        all_pos = np.vstack([X_t, head_X_t.reshape(1, 3)])
        all_forces = np.vstack([f_t, head_f_t.reshape(1, 3)])
        all_torques = np.vstack([n_t, head_n_t.reshape(1, 3)])
        
        # --- Grid setup ---
        grid_center = head_X_t
        grid_span = L_eff * 1.5
        grid_res = 60j
        Y, X = np.mgrid[grid_center[1]-grid_span:grid_center[1]+grid_span:grid_res,
                        grid_center[0]-grid_span:grid_center[0]+grid_span:grid_res]

        # --- Hydrodynamic Calculation ---
        # Epsilon should be on the order of the discretization length
        epsilon_filament = segment_length
        epsilon_head = R_head
        epsilon_vec = np.array([epsilon_filament] * M + [epsilon_head])

        U, V, magnitude = hydro_velocity_field(X, Y, all_pos, all_forces, all_pos, all_torques, epsilon_vec)
        
        # --- Plotting ---
        vmax = np.percentile(magnitude, 99)
        ax.imshow(magnitude.T, extent=[X.min(), X.max(), Y.min(), Y.max()],
                  origin='lower', cmap='viridis', vmax=vmax, alpha=0.8)
        ax.streamplot(X, Y, U, V, color='white', linewidth=0.7, density=1.5, arrowstyle='->')
        ax.plot(X_t[:, 0], X_t[:, 1], 'o-', lw=3, color='red', markersize=5, label='Filament')
        ax.plot(head_X_t[0], head_X_t[1], 'o', markersize=15, color='black', label='Head')
        
        # --- Formatting ---
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        ax.set_aspect('equal', adjustable='box')
        sim_time = frame * dt_snapshot
        ax.set_title(f"Swimmer and Fluid Flow (with Stokeslets & Rotlets) | Time: {sim_time:.3f} s")
        ax.legend(loc='upper right')
        ax.set_xlim(grid_center[0] - grid_span, grid_center[0] + grid_span)
        ax.set_ylim(grid_center[1] - grid_span, grid_center[1] + grid_span)
        
        print(f"Processing frame {frame+1}/{num_frames}", end='\r')

    # --- 7. Create and Save the Animation ---
    ani = FuncAnimation(
        fig,
        update_animation,
        frames=num_frames,
        interval=50
    )

    try:
        output_filename = 'swimmer_flow_field_with_torques.gif'
        print(f"\nSaving animation to '{output_filename}'...")
        ani.save(output_filename, writer='pillow', fps=20, dpi=100)
        print(f"Animation saved successfully!")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Please ensure you have 'pillow' and 'numba' installed (`pip install pillow numba`).")

    plt.show()

if __name__ == '__main__':
    create_flow_animation()
