import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba
import time
import os

# --- Parameters ---
PARAMS = {
    # Scenario setup
    "scenario": "flagellar_wave",  # "static_helix", "flagellar_wave"

    # --- Swimmer Geometry ---
    "head_radius": 1.3,  # Radius of the spherical head (um)
    "tail_attachment_pos": np.array([0, 0, -1.3]), # Attachment point on head surface (local coords)
    
    # Rod (Tail) properties
    "M": 100,  # Number of immersed boundary points for the tail
    "ds": 0.3,   # Meshwidth for rod (um)

    # Time integration
    "dt": 1.0e-6,  # Time step (s)
    "total_time": 0.05, # Total simulation time (s)

    # Fluid properties
    "mu": 1.0e-6,  # Fluid viscosity (g um^-1 s^-1)

    # Regularization
    "epsilon_reg_factor": 2.0, # Factor for regularization (epsilon_reg = epsilon_reg_factor * ds)

    # Material properties (g um^3 s^-2 for moduli)
    "a1": 3.5e-3, "a2": 3.5e-3, "a3": 3.5e-3, # Bending and twist moduli
    "b1": 8.0e-1, "b2": 8.0e-1, "b3": 8.0e-1, # Shear and stretch moduli
    
    # Intrinsic Curvature / Twist (for static_helix)
    "Omega1": 0.0, "Omega2": 0.0, "Omega3": 0.0,

    # --- FLAGELLAR WAVE PARAMETERS ---
    "k_wave": 2 * np.pi / 5.0, # Wave number (rad/um).
    "b_amp": 0.8,               # Wave amplitude (um).
    "sigma_freq": 275,          # Wave angular frequency (rad/s).

    # --- ANIMATION & OUTPUT SETTINGS ---
    "animation_interval": 50,
    "animation_steps_skip": 250,
    "save_history": False, # Set to True to save data files
    "flow_viz_resolution": 40, # Grid points for flow visualization (e.g., 40x40)
}

# --- Scenario-specific overrides ---
if PARAMS["scenario"] == "static_helix":
    PARAMS.update({
        "Omega1": 1.3, "Omega3": np.pi / 2.0,
        "total_time": 0.1,
    })

PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]

# --- Helper Functions for Regularized Stokes Flow ---
# (These functions remain the same as in your provided code)

@numba.njit(cache=True)
def H1_func(r, epsilon_reg):
    return (2*epsilon_reg**2 + r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

@numba.njit(cache=True)
def H2_func(r, epsilon_reg):
    return 1.0 / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

@numba.njit(cache=True)
def Q_func(r, epsilon_reg):
    return (5*epsilon_reg**2 + 2*r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(5.0/2.0))

@numba.njit(cache=True)
def D1_func(r, epsilon_reg):
    return (10*epsilon_reg**4 - 7*epsilon_reg**2*r**2 - 2*r**4) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))

@numba.njit(cache=True)
def D2_func(r, epsilon_reg):
    return (21*epsilon_reg**2 + 6*r**2) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))

# --- Rotation Helpers ---
def get_rotation_matrix_sqrt(R_mat):
    """Computes the principal square root of a 3x3 rotation matrix."""
    try:
        # Orthogonalize matrix to ensure it's a valid rotation matrix
        U, _, Vh = np.linalg.svd(R_mat)
        R_mat_ortho = U @ Vh
        if np.linalg.det(R_mat_ortho) < 0:
            Vh[-1,:] *= -1
            R_mat_ortho = U @ Vh
        
        rot = Rotation.from_matrix(R_mat_ortho)
        rotvec = rot.as_rotvec()
        sqrt_rotvec = rotvec * 0.5
        sqrt_R_mat = Rotation.from_rotvec(sqrt_rotvec).as_matrix()
        return sqrt_R_mat
    except Exception as e:
        # Fallback for identity or near-identity matrices
        if np.allclose(R_mat, np.eye(3)):
            return np.eye(3)
        raise e

def get_rodrigues_rotation_matrix(axis_vector, angle_rad):
    """Computes rotation matrix using Rodrigues' formula."""
    if np.isclose(angle_rad, 0):
        return np.eye(3)
    norm_axis = np.linalg.norm(axis_vector)
    if norm_axis < 1e-9 :
        return np.eye(3)
    axis_vector = axis_vector / norm_axis
    return Rotation.from_rotvec(axis_vector * angle_rad).as_matrix()

# --- Numba JITed core velocity computation ---
@numba.njit(cache=True)
def _compute_velocities_at_points_core(target_points, source_points, g0_sources, m0_sources, epsilon_reg, mu):
    """
    Numba-jitted core loop for computing velocities at arbitrary target points
    due to a set of Stokeslet and Rotlet sources.
    """
    num_targets = target_points.shape[0]
    num_sources = source_points.shape[0]
    u_out = np.zeros((num_targets, 3))
    w_out = np.zeros((num_targets, 3))

    for j in range(num_targets):
        Xj = target_points[j]
        sum_u_terms_j = np.zeros(3)
        sum_w_terms_j = np.zeros(3)

        for k in range(num_sources):
            Xk = source_points[k]
            g0k = g0_sources[k]
            m0k = m0_sources[k]
            
            r_vec_jk = Xj - Xk
            r_mag_jk_sq = r_vec_jk[0]**2 + r_vec_jk[1]**2 + r_vec_jk[2]**2
            r_mag_jk = np.sqrt(r_mag_jk_sq)

            if r_mag_jk < 1e-9:
                h1_val = H1_func(0.0, epsilon_reg)
                d1_val_self = D1_func(0.0, epsilon_reg)
                sum_u_terms_j += g0k * h1_val
                sum_w_terms_j += 0.25 * m0k * d1_val_self
            else:
                h1_val = H1_func(r_mag_jk, epsilon_reg)
                h2_val = H2_func(r_mag_jk, epsilon_reg)
                q_val = Q_func(r_mag_jk, epsilon_reg)
                d1_val = D1_func(r_mag_jk, epsilon_reg)
                d2_val = D2_func(r_mag_jk, epsilon_reg)

                # Linear velocity contributions
                dot_g0k_rvec = np.dot(g0k, r_vec_jk)
                term_uS = g0k * h1_val + dot_g0k_rvec * r_vec_jk * h2_val
                cross_m0k_rvec = np.cross(m0k, r_vec_jk)
                term_uR_for_u = 0.5 * cross_m0k_rvec * q_val
                sum_u_terms_j += term_uS + term_uR_for_u

                # Angular velocity contributions
                cross_g0k_rvec = np.cross(g0k, r_vec_jk)
                term_uR_for_w = 0.5 * cross_g0k_rvec * q_val
                dot_m0k_rvec = np.dot(m0k, r_vec_jk)
                term_uD = 0.25 * (m0k * d1_val + dot_m0k_rvec * r_vec_jk * d2_val)
                sum_w_terms_j += term_uR_for_w + term_uD
    
    u_out = sum_u_terms_j / mu
    w_out = sum_w_terms_j / mu
    
    return u_out, w_out

# --- Kirchhoff Rod Class (The Tail) ---
class KirchhoffRod:
    def __init__(self, params):
        self.p = params
        self.M = self.p["M"]
        self.ds = self.p["ds"]
        self.dt = self.p["dt"]
        
        # Material properties
        self.a = np.array([self.p["a1"], self.p["a2"], self.p["a3"]])
        self.b = np.array([self.p["b1"], self.p["b2"], self.p["b3"]])
        
        # State variables
        self.X = np.zeros((self.M, 3))
        self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3))
        self.D3 = np.zeros((self.M, 3))

        self.s_vals = np.arange(self.M) * self.ds
        omega_vec = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])
        self.Omega = np.tile(omega_vec, (self.M, 1))
        
        # Initialize as a straight rod along Z-axis
        self.X[:, 2] = self.s_vals
        self.D3[:, 2] = 1.0
        self.D1[:, 0] = 1.0
        self.D2[:, 1] = 1.0
        
    def _get_D_matrix(self, k):
        return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def update_intrinsic_curvature(self, time):
        if self.p["scenario"] == "flagellar_wave":
            k, b, sigma = self.p["k_wave"], self.p["b_amp"], self.p["sigma_freq"]
            self.Omega[:, 0] = -k**2 * b * np.sin(k * self.s_vals - sigma * time)
            self.Omega[:, 1] = 0.0
            self.Omega[:, 2] = 0.0

    def compute_internal_forces_and_moments(self):
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
        for k in range(self.M - 1):
            Dk_mat = self._get_D_matrix(k)
            Dkp1_mat = self._get_D_matrix(k + 1)
            Ak = Dkp1_mat @ Dk_mat.T
            try:
                sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            except Exception as e:
                print(f"Error computing sqrt of rotation matrix at rod segment {k}: {e}")
                raise e
            D_half_k_mat = sqrt_Ak @ Dk_mat
            D1_h, D2_h, D3_h = D_half_k_mat[0], D_half_k_mat[1], D_half_k_mat[2]

            dX_ds = (self.X[k+1] - self.X[k]) / self.ds
            F_coeffs = self.b * (np.array([np.dot(D1_h, dX_ds), np.dot(D2_h, dX_ds), np.dot(D3_h, dX_ds)]) - np.array([0,0,1]))
            F_half[k] = F_coeffs[0]*D1_h + F_coeffs[1]*D2_h + F_coeffs[2]*D3_h

            dD1ds, dD2ds, dD3ds = (self.D1[k+1]-self.D1[k])/self.ds, (self.D2[k+1]-self.D2[k])/self.ds, (self.D3[k+1]-self.D3[k])/self.ds
            N_coeffs = self.a * (np.array([np.dot(dD2ds, D3_h), np.dot(dD3ds, D1_h), np.dot(dD1ds, D2_h)]) - self.Omega[k,:])
            N_half[k] = N_coeffs[0]*D1_h + N_coeffs[1]*D2_h + N_coeffs[2]*D3_h
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        f_on_rod = np.zeros((self.M, 3))
        n_on_rod = np.zeros((self.M, 3))
        F_b_start, F_b_end = np.zeros(3), np.zeros(3) # Will be handled by head-tail interaction
        N_b_start, N_b_end = np.zeros(3), np.zeros(3) # Free end has zero torque
        
        for k in range(self.M):
            F_prev = F_b_start if k == 0 else F_half[k - 1]
            F_next = F_b_end if k == self.M - 1 else F_half[k]
            f_on_rod[k] = (F_next - F_prev) / self.ds

            N_prev = N_b_start if k == 0 else N_half[k - 1]
            N_next = N_b_end if k == self.M - 1 else N_half[k]
            dN_ds = (N_next - N_prev) / self.ds
            
            cross_term = np.zeros(3)
            if k < self.M - 1:
                cross_term += np.cross((self.X[k + 1] - self.X[k]) / self.ds, F_next)
            if k > 0:
                cross_term += np.cross((self.X[k] - self.X[k - 1]) / self.ds, F_prev)
            n_on_rod[k] = dN_ds + 0.5 * cross_term
        return f_on_rod, n_on_rod

    def update_state(self, u_rod, w_rod):
        self.X += u_rod * self.dt
        for k in range(self.M):
            wk = w_rod[k]
            wk_mag = np.linalg.norm(wk)
            if wk_mag > 1e-9:
                R_matrix = get_rodrigues_rotation_matrix(wk / wk_mag, wk_mag * self.dt)
                self.D1[k] = self.D1[k] @ R_matrix.T
                self.D2[k] = self.D2[k] @ R_matrix.T
                self.D3[k] = self.D3[k] @ R_matrix.T

# --- Full Swimmer Class ---
class Bacterium:
    def __init__(self, params):
        self.p = params
        self.mu = self.p["mu"]
        self.dt = self.p["dt"]
        
        # Head properties
        self.X_head = np.zeros(3)
        self.D_head = np.eye(3) # Orientation as a rotation matrix
        self.head_radius = self.p["head_radius"]
        self.head_epsilon_reg = self.head_radius # Regularization parameter for the head
        self.r_attach_local = self.p["tail_attachment_pos"] # Attachment vector in head's frame

        # Tail
        self.tail = KirchhoffRod(params)
        
        self.time = 0.0
        
        # Initial placement of the tail relative to the head
        self.enforce_attachment_constraint()

    def enforce_attachment_constraint(self):
        """Hard-sets the tail's position and orientation to match the head's attachment point."""
        # Position of the attachment point in the global frame
        X_attach_global = self.X_head + self.D_head @ self.r_attach_local
        
        # The tail's first point must be at this attachment point
        displacement = X_attach_global - self.tail.X[0]
        self.tail.X += displacement
        
        # The tail's orientation at the base should match the head's orientation
        # This aligns the material frame D of the tail with the body frame of the head.
        for i in range(3):
            self.tail.D1[0, i] = self.D_head[0, i]
            self.tail.D2[0, i] = self.D_head[1, i]
            self.tail.D3[0, i] = self.D_head[2, i]

    def get_all_sources(self, f_on_rod, n_on_rod):
        """Calculates head forces/torques and aggregates all sources for the hydrodynamic calculation."""
        # Total force and torque exerted by the tail on the fluid
        g0_tail = f_on_rod * self.tail.ds
        m0_tail = n_on_rod * self.tail.ds
        
        # Force/Torque Balance: The head must exert equal and opposite force/torque
        g0_head = -np.sum(g0_tail, axis=0)
        
        torque_from_tail_forces = np.sum(np.cross(self.tail.X - self.X_head, g0_tail), axis=0)
        torque_from_tail_moments = np.sum(m0_tail, axis=0)
        m0_head = -(torque_from_tail_forces + torque_from_tail_moments)

        # Aggregate all sources
        all_source_points = np.vstack([self.X_head, self.tail.X])
        all_g0 = np.vstack([g0_head, g0_tail])
        all_m0 = np.vstack([m0_head, m0_tail])
        # Use different regularization for head vs tail
        all_epsilons = np.concatenate([[self.head_epsilon_reg], [self.p["epsilon_reg"]] * self.tail.M])

        return all_source_points, all_g0, all_m0, all_epsilons

    def compute_all_velocities(self, all_sources):
        """Computes velocities for the head and every point on the tail."""
        all_source_points, all_g0, all_m0, all_epsilons = all_sources
        
        target_points = np.vstack([self.X_head, self.tail.X])
        
        # Simplified: Use a single epsilon for the numba call. A more complex implementation
        # could handle heterogeneous epsilons. Using the tail's epsilon is a reasonable approx.
        epsilon_for_calc = self.p["epsilon_reg"]

        # This part cannot be easily JITted with heterogeneous epsilons, so we do it in Python
        num_targets = target_points.shape[0]
        num_sources = all_source_points.shape[0]
        u_out = np.zeros((num_targets, 3))
        w_out = np.zeros((num_targets, 3))

        for j in range(num_targets):
            Xj = target_points[j]
            for k in range(num_sources):
                Xk = all_source_points[k]
                g0k, m0k = all_g0[k], all_m0[k]
                eps_k = all_epsilons[k]
                
                r_vec = Xj - Xk
                r_mag_sq = np.dot(r_vec, r_vec)
                r_mag = np.sqrt(r_mag_sq)

                if r_mag < 1e-9 and j==k:
                    # Self-term
                    u_out[j] += g0k * H1_func(0, eps_k)
                    w_out[j] += 0.25 * m0k * D1_func(0, eps_k)
                elif r_mag > 1e-9:
                    # Interaction term
                    u_out[j] += g0k * H1_func(r_mag, eps_k) + np.dot(g0k, r_vec) * r_vec * H2_func(r_mag, eps_k)
                    u_out[j] += 0.5 * np.cross(m0k, r_vec) * Q_func(r_mag, eps_k)
                    
                    w_out[j] += 0.5 * np.cross(g0k, r_vec) * Q_func(r_mag, eps_k)
                    w_out[j] += 0.25 * (m0k*D1_func(r_mag, eps_k) + np.dot(m0k,r_vec)*r_vec*D2_func(r_mag, eps_k))

        u_out /= self.mu
        w_out /= self.mu

        U_head, W_head = u_out[0], w_out[0]
        u_tail, w_tail = u_out[1:], w_out[1:]
        
        return U_head, W_head, u_tail, w_tail

    def update_head_state(self, U_head, W_head):
        """Updates head position and orientation."""
        self.X_head += U_head * self.dt
        W_mag = np.linalg.norm(W_head)
        if W_mag > 1e-9:
            R_update = get_rodrigues_rotation_matrix(W_head / W_mag, W_mag * self.dt)
            self.D_head = R_update @ self.D_head

    def simulation_step(self):
        # 1. Enforce physical connection between head and tail
        self.enforce_attachment_constraint()
        
        # 2. Update time-dependent intrinsic properties of the tail
        self.tail.update_intrinsic_curvature(self.time)
        
        # 3. Compute internal elastic forces/moments from the tail's current shape
        F_half, N_half = self.tail.compute_internal_forces_and_moments()
        
        # 4. Determine the fluid forces required to satisfy elasticity
        f_on_rod, n_on_rod = self.tail.compute_fluid_forces_on_rod(F_half, N_half)
        
        # 5. Determine head forces/torques via global balance and aggregate all sources
        all_sources = self.get_all_sources(f_on_rod, n_on_rod)
        
        # 6. Compute velocities of head and tail due to all hydrodynamic sources
        U_head, W_head, u_tail, w_tail = self.compute_all_velocities(all_sources)
        
        # 7. Update the state (position and orientation) of head and tail
        self.update_head_state(U_head, W_head)
        self.tail.update_state(u_tail, w_tail)
        
        # 8. Advance time
        self.time += self.dt
        
        return all_sources # Return sources for visualization

# --- Flow Field Visualization ---
def plot_flow_field(ax, bacterium, all_sources):
    """Calculates and plots the 2D flow field on the given axes."""
    ax.clear()
    
    # 1. Define the grid for visualization
    res = PARAMS["flow_viz_resolution"]
    plot_margin = bacterium.tail.p["L_eff"] * 0.5
    x_center, y_center = bacterium.X_head[0], bacterium.X_head[1]
    
    x_grid = np.linspace(x_center - plot_margin, x_center + plot_margin, res)
    y_grid = np.linspace(y_center - plot_margin, y_center + plot_margin, res)
    gx, gy = np.meshgrid(x_grid, y_grid)
    
    # Grid points in 3D (on the z=0 plane)
    grid_points_3d = np.vstack([gx.ravel(), gy.ravel(), np.zeros(res*res)]).T
    
    # 2. Calculate velocities on the grid
    # (Reusing the non-JITted velocity calculation from the Bacterium class)
    source_points, g0, m0, epsilons = all_sources
    num_grid_points = grid_points_3d.shape[0]
    num_sources = source_points.shape[0]
    grid_velocities = np.zeros_like(grid_points_3d)

    for j in range(num_grid_points):
        Xj = grid_points_3d[j]
        for k in range(num_sources):
            Xk, g0k, m0k, eps_k = source_points[k], g0[k], m0[k], epsilons[k]
            r_vec, r_mag = Xj - Xk, np.linalg.norm(Xj - Xk)
            if r_mag > 1e-9:
                grid_velocities[j] += g0k*H1_func(r_mag,eps_k) + np.dot(g0k,r_vec)*r_vec*H2_func(r_mag,eps_k)
                grid_velocities[j] += 0.5 * np.cross(m0k, r_vec) * Q_func(r_mag, eps_k)

    grid_velocities /= bacterium.mu
    
    # Reshape for plotting
    u_grid, v_grid = grid_velocities[:,0].reshape(res,res), grid_velocities[:,1].reshape(res,res)
    speed = np.sqrt(u_grid**2 + v_grid**2)

    # 3. Plotting
    # Color plot for speed
    ax.imshow(speed, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
              origin='lower', cmap='viridis', alpha=0.8)
    
    # Streamlines
    ax.streamplot(gx, gy, u_grid, v_grid, color='white', linewidth=0.7, density=1.5)
    
    # Plot the bacterium on top
    # Head
    head_circle = plt.Circle((bacterium.X_head[0], bacterium.X_head[1]), bacterium.head_radius, color='red', zorder=10)
    ax.add_patch(head_circle)
    # Tail
    ax.plot(bacterium.tail.X[:, 0], bacterium.tail.X[:, 1], '.-', color='black', lw=2, zorder=11)
    
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_title(f"Flow Field and Swimmer at t = {bacterium.time:.3e} s")
    ax.set_aspect('equal', adjustable='box')


# --- Main Simulation and Animation ---
if __name__ == '__main__':
    swimmer = Bacterium(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    
    history_states = []
    
    print(f"Starting simulation: {PARAMS['scenario']}")
    print(f"Total time: {PARAMS['total_time']}s, dt: {PARAMS['dt']}s -> {num_steps} steps.")
    start_time = time.time()

    for step in range(num_steps):
        all_sources = swimmer.simulation_step()
        
        if step % PARAMS["animation_steps_skip"] == 0:
            # Store a deep copy of the state for animation
            current_state = {
                'time': swimmer.time,
                'X_head': swimmer.X_head.copy(),
                'tail_X': swimmer.tail.X.copy(),
                'sources': all_sources
            }
            history_states.append(current_state)
            
            elapsed = time.time() - start_time
            print(f"Step {step}/{num_steps}, Sim Time: {swimmer.time:.2e} s, Wall Time: {elapsed:.2f}s")
            
            if np.isnan(swimmer.X_head).any() or np.isnan(swimmer.tail.X).any():
                print("ERROR: NaN detected. Simulation unstable.")
                break

    end_time = time.time()
    print(f"Simulation finished. Total wall time: {end_time - start_time:.2f} seconds.")

    # --- Animation Setup ---
    if not history_states:
        print("No history recorded. Exiting.")
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        def update_animation(frame_idx):
            state = history_states[frame_idx]
            
            # Create a dummy bacterium object just to pass state to the plotting function
            bacterium_for_plot = Bacterium(PARAMS)
            bacterium_for_plot.time = state['time']
            bacterium_for_plot.X_head = state['X_head']
            bacterium_for_plot.tail.X = state['tail_X']
            
            plot_flow_field(ax, bacterium_for_plot, state['sources'])

        ani = FuncAnimation(fig, update_animation, frames=len(history_states),
                            interval=PARAMS["animation_interval"])

        try:
            filename = f"bacterium_animation_{PARAMS['scenario']}.mp4"
            print(f"\nSaving animation to {filename}...")
            ani.save(filename, writer='ffmpeg', fps=20, dpi=150)
            print("Save complete.")
        except Exception as e:
            print(f"\nCould not save animation. Error: {e}")
            print("Please ensure ffmpeg is installed and accessible in your system's PATH.")
        
        plt.show()