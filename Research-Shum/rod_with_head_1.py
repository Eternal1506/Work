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
    "scenario": "flagellar_wave", 

    # --- NEW: Head and Visualization Controls ---
    "enable_head": True,            # If True, a spherical head is added.
    "head_radius": 2.0,             # Radius of the spherical head (um).

    "visualize_flow_field": True, # If True, generate flow field plots.
    "flow_vis_interval_steps": 250, # How often to generate a flow plot.
    "flow_vis_plane": 'yz',         # Plane for 2D flow visualization ('xy', 'xz', 'yz').
    
    # Rod properties
    "M": 60,
    "ds": 0.667, # Segment length (um)

    # Time integration
    "dt": 1.0e-6,
    "total_time": 0.06, # Shorter time for quicker test runs.

    # Fluid properties
    "mu": 1.0e-6,

    # Regularization
    "epsilon_reg_factor": 7.0,

    # Material properties (g um^3 s^-2 for moduli)
    "a1": 3.5e-3, "a2": 3.5e-3, "a3": 3.5e-3,
    "b1": 8.0e-1, "b2": 8.0e-1, "b3": 8.0e-1,

    "Omega1": 0.0, "Omega2": 0.0, "Omega3": 0.0,

    # --- INITIAL SHAPE ---
    "initial_shape": "straight",
    "straight_rod_orientation_axis": 'z',
    "xi_pert": 0,

    # --- FLAGELLAR WAVE PARAMETERS ---
    "k_wave": 2 * np.pi / 5.0,
    "b_amp": 0.2,
    "sigma_freq": 275,

    # --- ANIMATION & OUTPUT SETTINGS ---
    "animation_interval": 40,
    "animation_steps_skip": 100,
    "debugging": False,
    "debug_plot_interval_steps": 500,
    "save_history": True,
}

# --- Scenario-specific overrides ---
if PARAMS["scenario"] == "static_helix":
    PARAMS.update({
        "Omega1": 1.3, "Omega3": np.pi / 2.0,
        "initial_shape": "straight", "straight_rod_orientation_axis": 'z'
    })

PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]
if PARAMS["enable_head"]:
    # The head's regularization is its radius, as requested.
    PARAMS["epsilon_reg_head"] = PARAMS["head_radius"]


# --- Helper Functions for Regularized Stokes Flow (Unchanged) ---
@numba.njit(cache=True)
def H1_func(r, epsilon_reg): return (2*epsilon_reg**2 + r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))
@numba.njit(cache=True)
def H2_func(r, epsilon_reg): return 1.0 / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))
@numba.njit(cache=True)
def Q_func(r, epsilon_reg): return (5*epsilon_reg**2 + 2*r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(5.0/2.0))
@numba.njit(cache=True)
def D1_func(r, epsilon_reg): return (10*epsilon_reg**4 - 7*epsilon_reg**2*r**2 - 2*r**4) / (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))
@numba.njit(cache=True)
def D2_func(r, epsilon_reg): return (21*epsilon_reg**2 + 6*r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))


# --- Rotation Helpers (Unchanged) ---
def get_rotation_matrix_sqrt(R_mat):
    try:
        if not np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5):
            U, _, Vh = np.linalg.svd(R_mat)
            R_mat_ortho = U @ Vh
            if np.linalg.det(R_mat_ortho) < 0:
                Vh[-1,:] *= -1
                R_mat_ortho = U @ Vh
            R_mat = R_mat_ortho
        rot = Rotation.from_matrix(R_mat)
        sqrt_R_mat = Rotation.from_rotvec(rot.as_rotvec() * 0.5).as_matrix()
        return sqrt_R_mat
    except Exception as e:
        raise e

def get_rodrigues_rotation_matrix(axis_vector, angle_rad):
    if np.isclose(angle_rad, 0): return np.eye(3)
    norm_axis = np.linalg.norm(axis_vector)
    if norm_axis < 1e-9: return np.eye(3)
    return Rotation.from_rotvec(axis_vector / norm_axis * angle_rad).as_matrix()

# --- Numba JITed core velocity computation ---
@numba.njit(cache=True)
def _compute_velocities_on_body_core(X_eval, X_source, g0_source, m0_source, epsilon_source, mu):
    """
    Numba-jitted core loop for computing velocity AND angular velocity at evaluation points.
    This is used to find the velocity of the body points themselves.
    """
    num_eval = X_eval.shape[0]
    num_source = X_source.shape[0]
    u_out = np.zeros((num_eval, 3))
    w_out = np.zeros((num_eval, 3))

    for j in range(num_eval):
        Xj = X_eval[j]
        sum_u_terms_j = np.zeros(3)
        sum_w_terms_j = np.zeros(3)

        for k in range(num_source):
            Xk = X_source[k]
            g0k = g0_source[k]
            m0k = m0_source[k]
            epsilon_reg_k = epsilon_source[k]

            r_vec_jk = Xj - Xk
            r_mag_jk_sq = np.dot(r_vec_jk, r_vec_jk)
            
            if r_mag_jk_sq < 1e-18: # Self-term
                h1_val = H1_func(0.0, epsilon_reg_k)
                d1_val_self = D1_func(0.0, epsilon_reg_k)
                sum_u_terms_j += g0k * h1_val
                sum_w_terms_j += 0.25 * m0k * d1_val_self
            else:
                r_mag_jk = np.sqrt(r_mag_jk_sq)
                h1_val = H1_func(r_mag_jk, epsilon_reg_k)
                h2_val = H2_func(r_mag_jk, epsilon_reg_k)
                q_val = Q_func(r_mag_jk, epsilon_reg_k)
                d1_val = D1_func(r_mag_jk, epsilon_reg_k)
                d2_val = D2_func(r_mag_jk, epsilon_reg_k)

                # Linear velocity contributions
                term_uS = g0k * h1_val + np.dot(g0k, r_vec_jk) * r_vec_jk * h2_val
                term_uR_for_u = 0.5 * np.cross(m0k, r_vec_jk) * q_val
                sum_u_terms_j += term_uS + term_uR_for_u

                # Angular velocity contributions
                term_uR_for_w = 0.5 * np.cross(g0k, r_vec_jk) * q_val
                term_uD = 0.25 * (m0k * d1_val + np.dot(m0k, r_vec_jk) * r_vec_jk * d2_val)
                sum_w_terms_j += term_uR_for_w + term_uD

        u_out[j] = sum_u_terms_j / mu
        w_out[j] = sum_w_terms_j / mu
        
    return u_out, w_out

@numba.njit(cache=True)
def _compute_flow_field_core(X_eval, X_source, g0_source, m0_source, epsilon_source, mu):
    """
    Numba-jitted core loop for computing ONLY linear velocity at grid points.
    This is used for visualizing the external flow field.
    """
    num_eval = X_eval.shape[0]
    num_source = X_source.shape[0]
    u_out = np.zeros((num_eval, 3))

    for j in range(num_eval):
        Xj = X_eval[j]
        sum_u_terms_j = np.zeros(3)

        for k in range(num_source):
            Xk = X_source[k]
            g0k = g0_source[k]
            m0k = m0_source[k]
            epsilon_reg_k = epsilon_source[k]

            r_vec_jk = Xj - Xk
            r_mag_jk_sq = np.dot(r_vec_jk, r_vec_jk)
            
            if r_mag_jk_sq < 1e-18:
                continue # Skip evaluation at the source point itself
            
            r_mag_jk = np.sqrt(r_mag_jk_sq)
            h1_val = H1_func(r_mag_jk, epsilon_reg_k)
            h2_val = H2_func(r_mag_jk, epsilon_reg_k)
            q_val = Q_func(r_mag_jk, epsilon_reg_k)

            # Linear velocity contributions
            term_uS = g0k * h1_val + np.dot(g0k, r_vec_jk) * r_vec_jk * h2_val
            term_uR_for_u = 0.5 * np.cross(m0k, r_vec_jk) * q_val
            sum_u_terms_j += term_uS + term_uR_for_u

        u_out[j] = sum_u_terms_j / mu
        
    return u_out

# --- KirchhoffRod Class (Modified slightly for clarity) ---
class KirchhoffRod:
    def __init__(self, params):
        self.p = params
        self.M = self.p["M"]; self.ds = self.p["ds"]; self.mu = self.p["mu"]
        self.epsilon_reg = self.p["epsilon_reg"]; self.dt = self.p["dt"]; self.L_eff = self.p["L_eff"]
        self.a = np.array([self.p["a1"],self.p["a2"],self.p["a3"]])
        self.b = np.array([self.p["b1"],self.p["b2"],self.p["b3"]])
        
        self.X = np.zeros((self.M, 3)); self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3)); self.D3 = np.zeros((self.M, 3))
        self.s_vals = np.arange(self.M) * self.ds
        omega_vec = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])
        self.Omega = np.tile(omega_vec, (self.M, 1))
        
        # --- Initial Shape Setup ---
        initial_shape = self.p.get("initial_shape", "straight")
        if initial_shape == "straight":
            orient_axis = self.p.get("straight_rod_orientation_axis", 'z')
            if orient_axis == 'x': self.X[:, 0] = self.s_vals; self.D3[:, 0] = 1.0; self.D1[:, 1] = 1.0; self.D2[:, 2] = 1.0
            elif orient_axis == 'y': self.X[:, 1] = self.s_vals; self.D3[:, 1] = 1.0; self.D1[:, 2] = 1.0; self.D2[:, 0] = 1.0
            else: self.X[:, 2] = self.s_vals; self.D3[:, 2] = 1.0; self.D1[:, 0] = 1.0; self.D2[:, 1] = 1.0
            
            pert_rot = get_rodrigues_rotation_matrix(self.D3[0], self.p["xi_pert"])
            self.D1 = self.D1 @ pert_rot.T; self.D2 = self.D2 @ pert_rot.T
            
        for i in range(self.M): # Orthonormalize directors
            d1, d2 = self.D1[i].copy(), self.D2[i].copy()
            norm_d1 = np.linalg.norm(d1)
            self.D1[i] = d1 / norm_d1 if norm_d1 > 1e-9 else np.array([1.0,0.0,0.0])
            d2_ortho = d2 - np.dot(d2, self.D1[i]) * self.D1[i]
            norm_d2_ortho = np.linalg.norm(d2_ortho)
            if norm_d2_ortho < 1e-9:
                temp_vec = np.array([0.0,1.0,0.0]) if np.abs(self.D1[i,0]) > 0.9 else np.array([1.0,0.0,0.0])
                self.D2[i] = np.cross(self.D3[i], self.D1[i])
            else: self.D2[i] = d2_ortho / norm_d2_ortho
            self.D3[i] = np.cross(self.D1[i], self.D2[i])

    def _get_D_matrix(self, k): return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def update_intrinsic_curvature(self, time):
        if self.p["scenario"] == "flagellar_wave":
            k, b, sigma = self.p["k_wave"], self.p["b_amp"], self.p["sigma_freq"]
            self.Omega[:, 0] = -k**2 * b * np.sin(k * self.s_vals + sigma * time)
            self.Omega[:, 1:] = 0.0

    def compute_internal_forces_and_moments(self):
        F_half = np.zeros((self.M - 1, 3)); N_half = np.zeros((self.M - 1, 3))
        for k in range(self.M - 1):
            Dk_mat, Dkp1_mat = self._get_D_matrix(k), self._get_D_matrix(k+1)
            Ak = Dkp1_mat @ Dk_mat.T
            try: sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            except Exception as e: raise e
            D_half_k_mat = sqrt_Ak @ Dk_mat
            D1_h, D2_h, D3_h = D_half_k_mat[0], D_half_k_mat[1], D_half_k_mat[2]
            
            dX_ds = (self.X[k+1] - self.X[k]) / self.ds
            F_coeffs = self.b * (np.array([np.dot(D1_h, dX_ds), np.dot(D2_h, dX_ds), np.dot(D3_h, dX_ds)]) - np.array([0,0,1]))
            F_half[k] = F_coeffs[0]*D1_h + F_coeffs[1]*D2_h + F_coeffs[2]*D3_h

            dD1ds,dD2ds,dD3ds = (self.D1[k+1]-self.D1[k])/self.ds, (self.D2[k+1]-self.D2[k])/self.ds, (self.D3[k+1]-self.D3[k])/self.ds
            N_coeffs = self.a * (np.array([np.dot(dD2ds,D3_h),np.dot(dD3ds,D1_h),np.dot(dD1ds,D2_h)])-self.Omega[k])
            N_half[k] = N_coeffs[0]*D1_h + N_coeffs[1]*D2_h + N_coeffs[2]*D3_h
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        f_on_rod=np.zeros((self.M,3)); n_on_rod=np.zeros((self.M,3))
        F_b_start, F_b_end, N_b_start, N_b_end = F_half[0], np.zeros(3), N_half[0], np.zeros(3)
        # Note: The force/moment at the start are now determined by the head interaction, not zero.
        # This will be handled in the Swimmer class. Here we just calculate the derivatives.

        for k in range(1, self.M-1): # Central difference for internal points
             f_on_rod[k] = (F_half[k] - F_half[k-1])/self.ds
             dN_ds = (N_half[k] - N_half[k-1])/self.ds
             cross_term = 0.5 * (np.cross((self.X[k+1]-self.X[k])/self.ds, F_half[k]) + np.cross((self.X[k]-self.X[k-1])/self.ds, F_half[k-1]))
             n_on_rod[k] = dN_ds + cross_term
        
        # End points (boundary conditions)
        f_on_rod[0] = (F_half[0] - F_b_start)/self.ds # Placeholder, will be determined by head
        f_on_rod[-1] = (F_b_end - F_half[-1])/self.ds
        n_on_rod[0] = ((N_half[0] - N_b_start)/self.ds + 0.5 * np.cross((self.X[1]-self.X[0])/self.ds, F_half[0])) # Placeholder
        n_on_rod[-1] = ((N_b_end - N_half[-1])/self.ds + 0.5 * np.cross((self.X[-1]-self.X[-2])/self.ds, F_half[-1]))

        return f_on_rod, n_on_rod

# --- NEW: Swimmer Class ---
# --- NEW: Swimmer Class (Corrected for Stability) ---
class Swimmer:
    def __init__(self, params):
        self.p = params
        self.dt = self.p['dt']
        self.time = 0.0
        self.rod = KirchhoffRod(params) # The tail

        # Head state variables
        self.X_head = np.zeros(3)
        self.D1_head, self.D2_head, self.D3_head = np.eye(3) # Initial orientation
        
        # Define attachment point in head's local frame. Flagellum attaches at the "back".
        if self.p['enable_head']:
            self.X_attach_local = np.array([0, 0, -self.p['head_radius']])
        
            # Initial placement of the swimmer
            self.X_head = np.array([0.0, 0.0, 0.0]) # Start head at the origin
            self.D1_head, self.D2_head, self.D3_head = np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])
            
            # Position the tail relative to the head
            D_head_mat = np.array([self.D1_head, self.D2_head, self.D3_head])
            X_attach_global = self.X_head + D_head_mat.T @ self.X_attach_local
            
            # Shift the entire pre-calculated rod to connect to the attachment point
            rod_start_pos = self.rod.X[0].copy()
            self.rod.X += (X_attach_global - rod_start_pos)
    
    def get_head_D_matrix(self):
        return np.array([self.D1_head, self.D2_head, self.D3_head])

    def simulation_step(self):
        # 1. Update tail's intrinsic properties
        self.rod.update_intrinsic_curvature(self.time)

        # 2. Compute internal elastic forces/moments in the tail
        F_half, N_half = self.rod.compute_internal_forces_and_moments()

        # 3. CORRECTED: Compute forces/torques the swimmer exerts ON THE FLUID (g0, m0)
        # These are the source terms for the Stokeslet calculation.
        
        g0_head = np.zeros(3)
        m0_head = np.zeros(3)
        g0_rod = np.zeros((self.rod.M, 3))
        m0_rod = np.zeros((self.rod.M, 3))

        if self.p['enable_head']:
            # The head must exert a force on the fluid to balance the pull from the tail.
            # The tail pulls the head with force -F_half[0].
            # So, the head must push the fluid with force +F_half[0].
            g0_head = F_half[0]

            # The head must also exert a torque on the fluid to balance the moment from the tail.
            # The tail exerts moment -N_half[0] and a moment from the force cross(r, -F_half[0]).
            # The head must exert the opposite torque on the fluid to stay in balance.
            X_attach_global = self.rod.X[0]
            r_vec = X_attach_global - self.X_head
            m0_head = N_half[0] + np.cross(r_vec, F_half[0])

        # The tail exerts forces on the fluid from the divergence of its internal stress.
        # Force density on fluid: g = dF/ds. Point force: g_k = F_{k+1/2} - F_{k-1/2}
        # Torque density on fluid: m = dN/ds + (dX/ds x F). Point torque: m_k = (N_{k+1/2}-N_{k-1/2}) + ...
        
        # Point k=0 (attachment point)
        # Force from head connection is F_half[0]. So g0_rod[0] = F_half[0] - F_half[0] = 0.
        # The force is fully transferred to the head's Stokeslet.
        g0_rod[0] = np.zeros(3)
        # For torque, the dN/ds term is also zero. We only have the cross product part.
        t_half_ds = (self.rod.X[1] - self.rod.X[0]) # This is tangent * ds
        m0_rod[0] = 0.5 * np.cross(t_half_ds, F_half[0])

        # Internal points k=1 to M-2
        for k in range(1, self.rod.M - 1):
            g0_rod[k] = F_half[k] - F_half[k-1]
            
            dN = N_half[k] - N_half[k-1]
            t_plus_ds = (self.rod.X[k+1] - self.rod.X[k])
            t_minus_ds = (self.rod.X[k] - self.rod.X[k-1])
            cross_term = 0.5 * (np.cross(t_plus_ds, F_half[k]) + np.cross(t_minus_ds, F_half[k-1]))
            m0_rod[k] = dN + cross_term

        # End point k=M-1 (free end, F=0, N=0 at the tip)
        if self.rod.M > 1:
            g0_rod[-1] = 0.0 - F_half[-1] # F_half[-1] is F_{M-3/2}
            
            dN_end = 0.0 - N_half[-1]
            t_minus_ds_end = (self.rod.X[-1] - self.rod.X[-2])
            cross_term_end = 0.5 * np.cross(t_minus_ds_end, F_half[-1])
            m0_rod[-1] = dN_end + cross_term_end


        # 4. Assemble all sources for velocity calculation
        if self.p['enable_head']:
            X_source = np.vstack([self.X_head.reshape(1,3), self.rod.X])
            g0_source = np.vstack([g0_head.reshape(1,3), g0_rod])
            m0_source = np.vstack([m0_head.reshape(1,3), m0_rod])
            epsilon_source = np.array([self.p['epsilon_reg_head']] + [self.p['epsilon_reg']] * self.rod.M)
        else:
            X_source = self.rod.X
            g0_source = g0_rod
            m0_source = m0_rod
            epsilon_source = np.array([self.p['epsilon_reg']] * self.rod.M)
        
        X_eval = X_source # Evaluate velocity at all body points

        # 5. Compute velocities of the body
        u_all, w_all = _compute_velocities_on_body_core(X_eval, X_source, g0_source, m0_source, epsilon_source, self.p['mu'])

        # 6. Update state (position and orientation)
        if self.p['enable_head']:
            u_head, w_head = u_all[0], w_all[0]
            u_rod, w_rod = u_all[1:], w_all[1:]

            # Update head
            self.X_head += u_head * self.dt
            rot_mat_head = get_rodrigues_rotation_matrix(w_head, self.dt)
            self.D1_head = self.D1_head @ rot_mat_head.T
            self.D2_head = self.D2_head @ rot_mat_head.T
            self.D3_head = self.D3_head @ rot_mat_head.T
        else:
            u_rod, w_rod = u_all, w_all

        # Update tail
        self.rod.X += u_rod * self.dt
        for k in range(self.rod.M):
            rot_mat_rod = get_rodrigues_rotation_matrix(w_rod[k], self.dt)
            self.rod.D1[k] = self.rod.D1[k] @ rot_mat_rod.T
            self.rod.D2[k] = self.rod.D2[k] @ rot_mat_rod.T
            self.rod.D3[k] = self.rod.D3[k] @ rot_mat_rod.T

        # 7. Enforce rigid connection from head to tail
        if self.p['enable_head']:
            D_head_mat = self.get_head_D_matrix()
            self.rod.X[0] = self.X_head + D_head_mat.T @ self.X_attach_local
            # Optional: could also align rod.D_k[0] with head frame

        self.time += self.dt
        # Return the forces on the fluid for visualization
        return g0_source, m0_source, X_source, epsilon_source

# --- NEW: Flow Field Visualization Function ---
def visualize_and_save_flow_field(swimmer, forces, torques, sources, epsilons, step, output_dir="flow_visualization"):
    """Generates and saves a 2D plot of the fluid flow field."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the 2D grid for visualization
    if swimmer.p['enable_head']:
        center = swimmer.X_head
    else:
        center = np.mean(swimmer.rod.X, axis=0)
    
    plot_range = swimmer.rod.L_eff * 1.5
    
    plane = swimmer.p['flow_vis_plane']
    if plane == 'xy':
        x = np.linspace(center[0] - plot_range, center[0] + plot_range, 50)
        y = np.linspace(center[1] - plot_range, center[1] + plot_range, 50)
        X_grid, Y_grid = np.meshgrid(x, y)
        Z_grid = np.full_like(X_grid, center[2])
        eval_points = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T
        ax_labels = ['X (um)', 'Y (um)']
    elif plane == 'xz':
        x = np.linspace(center[0] - plot_range, center[0] + plot_range, 50)
        z = np.linspace(center[2] - plot_range, center[2] + plot_range, 50)
        X_grid, Z_grid = np.meshgrid(x, z)
        Y_grid = np.full_like(X_grid, center[1])
        eval_points = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T
        ax_labels = ['X (um)', 'Z (um)']
    else: # yz
        y = np.linspace(center[1] - plot_range, center[1] + plot_range, 50)
        z = np.linspace(center[2] - plot_range, center[2] + plot_range, 50)
        Y_grid, Z_grid = np.meshgrid(y, z)
        X_grid = np.full_like(Y_grid, center[0])
        eval_points = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T
        ax_labels = ['Y (um)', 'Z (um)']

    # Compute velocities on the grid
    u_flow = _compute_flow_field_core(eval_points, sources, forces, torques, epsilons, swimmer.p['mu'])
    
    if plane == 'xy':
        U_flow = u_flow[:, 0].reshape(X_grid.shape)
        V_flow = u_flow[:, 1].reshape(Y_grid.shape)
        plot_grid_1, plot_grid_2 = X_grid, Y_grid
    elif plane == 'xz':
        U_flow = u_flow[:, 0].reshape(X_grid.shape)
        V_flow = u_flow[:, 2].reshape(Z_grid.shape)
        plot_grid_1, plot_grid_2 = X_grid, Z_grid
    else: # yz
        U_flow = u_flow[:, 1].reshape(Y_grid.shape)
        V_flow = u_flow[:, 2].reshape(Z_grid.shape)
        plot_grid_1, plot_grid_2 = Y_grid, Z_grid
        
    speed = np.sqrt(U_flow**2 + V_flow**2)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color plot for velocity magnitude
    c = ax.pcolormesh(plot_grid_1, plot_grid_2, speed, cmap='viridis', shading='gouraud', alpha=0.7)
    fig.colorbar(c, ax=ax, label='Velocity Magnitude (um/s)')
    
    # Streamlines for velocity direction
    ax.streamplot(plot_grid_1, plot_grid_2, U_flow, V_flow, color='black', linewidth=1, density=1.5)
    
    # Superimpose swimmer
    rod_X = swimmer.rod.X
    if plane == 'xy':
        ax.plot(rod_X[:,0], rod_X[:,1], 'o-', color='red', lw=3, markersize=4, label='Flagellum')
        if swimmer.p['enable_head']:
            head_circle = plt.Circle((swimmer.X_head[0], swimmer.X_head[1]), swimmer.p['head_radius'], color='darkred', zorder=10)
            ax.add_artist(head_circle)
    elif plane == 'xz':
        ax.plot(rod_X[:,0], rod_X[:,2], 'o-', color='red', lw=3, markersize=4, label='Flagellum')
        if swimmer.p['enable_head']:
            head_circle = plt.Circle((swimmer.X_head[0], swimmer.X_head[2]), swimmer.p['head_radius'], color='darkred', zorder=10)
            ax.add_artist(head_circle)
    else: # yz
        ax.plot(rod_X[:,1], rod_X[:,2], 'o-', color='red', lw=3, markersize=4, label='Flagellum')
        if swimmer.p['enable_head']:
            head_circle = plt.Circle((swimmer.X_head[1], swimmer.X_head[2]), swimmer.p['head_radius'], color='darkred', zorder=10)
            ax.add_artist(head_circle)
    
    ax.set_xlabel(ax_labels[0]); ax.set_ylabel(ax_labels[1])
    ax.set_title(f"Flow Field at t = {swimmer.time:.4f} s (Plane: {plane.upper()})")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"flow_field_step_{step:05d}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  -> Saved flow field visualization: {filename}")


# --- Main Simulation and Animation ---
if __name__ == '__main__':
    swimmer = Swimmer(PARAMS)
    rod = swimmer.rod # Convenience reference to the tail
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    
    history_X = [] # List to store rod centerline positions for animation

    print(f"Starting simulation for {PARAMS['total_time']}s ({num_steps} steps).")
    print(f"Head enabled: {PARAMS['enable_head']}")
    if PARAMS['enable_head']: print(f"Head radius: {PARAMS['head_radius']:.2f} um")
    print(f"Flow visualization enabled: {PARAMS['visualize_flow_field']}")

    start_time = time.time()
    for step in range(num_steps):
        g0, m0, sources, epsilons = swimmer.simulation_step()
            
        # Store data for animation
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            if step % (PARAMS["animation_steps_skip"]*10) == 0:
                elapsed_wall_time = time.time() - start_time
                print(f"Step {step}/{num_steps}, Sim Time: {swimmer.time:.2e} s, Wall Time: {elapsed_wall_time:.2f}s.")

        # Generate and save flow field plot at specified intervals
        if PARAMS["visualize_flow_field"] and step % PARAMS["flow_vis_interval_steps"] == 0:
            visualize_and_save_flow_field(swimmer, g0, m0, sources, epsilons, step)
            
        # Stability checks
        if np.max(np.abs(rod.X)) > rod.L_eff * 50 or np.isnan(rod.X).any():
            print("Simulation unstable. Coordinates exploded or became NaN.")
            break

    end_time = time.time()
    print(f"Simulation finished. Total wall time: {end_time - start_time:.2f} seconds.")

    # --- 3D Animation of the swimmer ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=3, color='blue', label='Flagellum')
    head_plot, = ax.plot([], [], [], 'o', markersize=10, color='red', label='Head') if PARAMS['enable_head'] else (None,)

    if history_X:
        all_coords = np.concatenate(history_X, axis=0)
        plot_range = np.max(np.ptp(all_coords, axis=0)) * 0.6
        if plot_range < 1e-6: plot_range = rod.L_eff or 1.0
        center = np.mean(all_coords, axis=0)
        ax.set_xlim([center[0]-plot_range, center[0]+plot_range])
        ax.set_ylim([center[1]-plot_range, center[1]+plot_range])
        ax.set_zlim([center[2]-plot_range, center[2]+plot_range])
        try: ax.set_aspect('equal', adjustable='box')
        except NotImplementedError: pass

    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)"); ax.set_zlabel("Z (um)")
    ax.set_title("Swimmer Dynamics")
    ax.legend()
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def init_animation():
        line.set_data_3d([], [], [])
        if head_plot: head_plot.set_data_3d([], [], [])
        time_text.set_text('')
        return [line, head_plot, time_text] if head_plot else [line, time_text]

    def update_animation(frame_idx):
        X_data = history_X[frame_idx]
        line.set_data_3d(X_data[:,0], X_data[:,1], X_data[:,2])
        if PARAMS['enable_head']:
            # For simplicity, we approximate head position from the first point of the saved tail history
            # A more robust way would be to save head history separately.
            head_pos = X_data[0] + swimmer.X_attach_local # Approximate
            head_plot.set_data_3d([head_pos[0]], [head_pos[1]], [head_pos[2]])
        
        current_time = frame_idx * PARAMS["animation_steps_skip"] * PARAMS["dt"]
        time_text.set_text(f'Time: {current_time:.3e} s')
        return [line, head_plot, time_text] if head_plot else [line, time_text]

    if history_X:
        ani = FuncAnimation(fig, update_animation, frames=len(history_X),
                            init_func=init_animation, blit=False, interval=PARAMS["animation_interval"])
        try:
            save_filename = 'swimmer_animation.mp4'
            print(f"Attempting to save 3D animation as {save_filename}...")
            ani.save(save_filename, writer='ffmpeg', fps=15, dpi=150)
            print(f"Animation saved as {save_filename}")
        except Exception as e:
            print(f"Error saving as MP4: {e}. Make sure ffmpeg is installed.")
        
        plt.tight_layout(); plt.show()
    else: 
        print("No history data to animate.")