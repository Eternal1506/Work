import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba # Added Numba import
import time # For timing
import os

# --- Parameters ---
PARAMS = {
    # Scenario setup
    "scenario": "flagellar_wave", 

    # Rod properties
    "M": 150,  # Number of immersed boundary points
    "ds": 0.087,  # Meshwidth for rod (um)

    # Time integration
    "dt": 5.0e-7,  # Time step (s) - Reduced for better stability
    "total_time": 0.05, # Total simulation time (s) - Reduced for testing

    # Fluid properties
    "mu": 1.0e-6,  # Fluid viscosity (g um^-1 s^-1)

    # Regularization
    "epsilon_reg_factor": 7.0, # Factor for regularization parameter (epsilon_reg = epsilon_reg_factor * ds)

    # Material properties (g um^3 s^-2 for moduli)
    "a1": 3.5e-3,  # Bending modulus
    "a2": 3.5e-3,  # Bending modulus
    "a3": 3.5e-3,  # Twist modulus
    "b1": 8.0e-1,  # Shear modulus
    "b2": 8.0e-1,  # Shear modulus
    "b3": 8.0e-1,  # Stretch modulus

    "Omega1": 0.0, # Intrinsic curvature in D1 direction
    "Omega2": 0.0, # Intrinsic curvature in D2 direction
    "Omega3": 0.0, # Intrinsic twist

    # --- INITIAL SHAPE ---
    "initial_shape": "straight", 
    "straight_rod_orientation_axis": 'z', 
    "xi_pert": 0,

    # --- FLAGELLAR WAVE PARAMETERS ---
    "k_wave": 2 * np.pi / 5.0, 
    "b_amp": 0.8,               
    "sigma_freq": 275,           

    # --- HEAD PARAMETERS ---
    "include_head": True,        # Whether to include spherical head
    "head_radius": 0.3,         # Radius of spherical head (um) - Reduced for stability
    "head_attachment_offset": 0.7, # Distance from head center to flagellum attachment (fraction of radius)

    # --- FLOW VISUALIZATION PARAMETERS ---
    "visualize_flow": True,      # Whether to visualize flow field
    "flow_grid_size": 20,        # Grid size for flow visualization (NxN grid)
    "flow_grid_extent": 8.0,     # Extent of flow grid in microns
    "flow_arrow_scale": 0.3,     # Scale factor for flow arrows
    "flow_color_velocity": True, # Color arrows by velocity magnitude

    # --- ANIMATION SETTINGS ---
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

# --- Helper Functions for Regularized Stokes Flow ---

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

@numba.njit(cache=True)
def stokeslet_velocity(x_eval, x_source, f_source, mu, epsilon_reg):
    """
    Compute velocity at x_eval due to a regularized Stokeslet at x_source with force f_source.
    This is just the translational part (H1 and H2 terms).
    """
    r_vec = x_eval - x_source
    r_mag = np.sqrt(np.sum(r_vec**2))
    
    if r_mag < 1e-12:
        h1_val = H1_func(0.0, epsilon_reg)
        return f_source * h1_val / mu
    else:
        h1_val = H1_func(r_mag, epsilon_reg)
        h2_val = H2_func(r_mag, epsilon_reg)
        
        term1 = f_source * h1_val
        term2 = np.dot(f_source, r_vec) * r_vec * h2_val
        return (term1 + term2) / mu

# --- Rotation Helpers ---
def get_rotation_matrix_sqrt(R_mat):
    """Computes the principal square root of a 3x3 rotation matrix."""
    try:
        if not np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5):
            U, _, Vh = np.linalg.svd(R_mat)
            R_mat_ortho = U @ Vh
            if np.linalg.det(R_mat_ortho) < 0:
                Vh[-1,:] *= -1
                R_mat_ortho = U @ Vh
            R_mat = R_mat_ortho 

        rot = Rotation.from_matrix(R_mat)
        rotvec = rot.as_rotvec()
        sqrt_rotvec = rotvec * 0.5
        sqrt_R_mat = Rotation.from_rotvec(sqrt_rotvec).as_matrix()
        return sqrt_R_mat
    except Exception as e:
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
def _compute_velocities_core_with_head(M_rod, X_rod, g0_all_rod, m0_all_rod, 
                                      X_head, g0_head, m0_head,
                                      epsilon_reg_rod, epsilon_reg_head, mu_rod):
    """
    Numba-jitted core loop for velocity computation including head contribution.
    """
    u_rod_out = np.zeros((M_rod, 3)) 
    w_rod_out = np.zeros((M_rod, 3))
    u_head_out = np.zeros(3)
    w_head_out = np.zeros(3)

    # Compute velocities at each rod point
    for j in range(M_rod): 
        Xj = X_rod[j]
        sum_u_terms_j = np.zeros(3)
        sum_w_terms_j = np.zeros(3)

        # Contribution from other rod points
        for k in range(M_rod): 
            Xk = X_rod[k]
            g0k = g0_all_rod[k] 
            m0k = m0_all_rod[k] 
            
            r_vec_jk = np.empty(3)
            r_vec_jk[0] = Xj[0] - Xk[0]
            r_vec_jk[1] = Xj[1] - Xk[1]
            r_vec_jk[2] = Xj[2] - Xk[2]
            
            r_mag_jk_sq = r_vec_jk[0]**2 + r_vec_jk[1]**2 + r_vec_jk[2]**2
            r_mag_jk = np.sqrt(r_mag_jk_sq)

            if r_mag_jk < 1e-9: 
                h1_val = H1_func(0.0, epsilon_reg_rod)
                d1_val_self = D1_func(0.0, epsilon_reg_rod)
                
                sum_u_terms_j[0] += g0k[0] * h1_val
                sum_u_terms_j[1] += g0k[1] * h1_val
                sum_u_terms_j[2] += g0k[2] * h1_val

                sum_w_terms_j[0] += 0.25 * m0k[0] * d1_val_self
                sum_w_terms_j[1] += 0.25 * m0k[1] * d1_val_self
                sum_w_terms_j[2] += 0.25 * m0k[2] * d1_val_self
            else:
                h1_val = H1_func(r_mag_jk, epsilon_reg_rod)
                h2_val = H2_func(r_mag_jk, epsilon_reg_rod)
                q_val = Q_func(r_mag_jk, epsilon_reg_rod)
                d1_val = D1_func(r_mag_jk, epsilon_reg_rod)
                d2_val = D2_func(r_mag_jk, epsilon_reg_rod)

                dot_g0k_rvec = g0k[0]*r_vec_jk[0] + g0k[1]*r_vec_jk[1] + g0k[2]*r_vec_jk[2]
                term_uS_0 = g0k[0] * h1_val + dot_g0k_rvec * r_vec_jk[0] * h2_val
                term_uS_1 = g0k[1] * h1_val + dot_g0k_rvec * r_vec_jk[1] * h2_val
                term_uS_2 = g0k[2] * h1_val + dot_g0k_rvec * r_vec_jk[2] * h2_val
                
                cross_m0k_rvec_0 = m0k[1]*r_vec_jk[2] - m0k[2]*r_vec_jk[1]
                cross_m0k_rvec_1 = m0k[2]*r_vec_jk[0] - m0k[0]*r_vec_jk[2]
                cross_m0k_rvec_2 = m0k[0]*r_vec_jk[1] - m0k[1]*r_vec_jk[0]
                term_uR_for_u_0 = 0.5 * cross_m0k_rvec_0 * q_val
                term_uR_for_u_1 = 0.5 * cross_m0k_rvec_1 * q_val
                term_uR_for_u_2 = 0.5 * cross_m0k_rvec_2 * q_val

                sum_u_terms_j[0] += term_uS_0 + term_uR_for_u_0
                sum_u_terms_j[1] += term_uS_1 + term_uR_for_u_1
                sum_u_terms_j[2] += term_uS_2 + term_uR_for_u_2

                cross_g0k_rvec_0 = g0k[1]*r_vec_jk[2] - g0k[2]*r_vec_jk[1]
                cross_g0k_rvec_1 = g0k[2]*r_vec_jk[0] - g0k[0]*r_vec_jk[2]
                cross_g0k_rvec_2 = g0k[0]*r_vec_jk[1] - g0k[1]*r_vec_jk[0]
                term_uR_for_w_0 = 0.5 * cross_g0k_rvec_0 * q_val
                term_uR_for_w_1 = 0.5 * cross_g0k_rvec_1 * q_val
                term_uR_for_w_2 = 0.5 * cross_g0k_rvec_2 * q_val
                
                dot_m0k_rvec = m0k[0]*r_vec_jk[0] + m0k[1]*r_vec_jk[1] + m0k[2]*r_vec_jk[2]
                term_uD_0 = 0.25 * m0k[0] * d1_val + 0.25 * dot_m0k_rvec * r_vec_jk[0] * d2_val
                term_uD_1 = 0.25 * m0k[1] * d1_val + 0.25 * dot_m0k_rvec * r_vec_jk[1] * d2_val
                term_uD_2 = 0.25 * m0k[2] * d1_val + 0.25 * dot_m0k_rvec * r_vec_jk[2] * d2_val

                sum_w_terms_j[0] += term_uR_for_w_0 + term_uD_0
                sum_w_terms_j[1] += term_uR_for_w_1 + term_uD_1
                sum_w_terms_j[2] += term_uR_for_w_2 + term_uD_2

        # Contribution from head
        r_vec_jh = np.empty(3)
        r_vec_jh[0] = Xj[0] - X_head[0]
        r_vec_jh[1] = Xj[1] - X_head[1]
        r_vec_jh[2] = Xj[2] - X_head[2]
        
        r_mag_jh = np.sqrt(r_vec_jh[0]**2 + r_vec_jh[1]**2 + r_vec_jh[2]**2)

        if r_mag_jh < 1e-9:
            h1_val = H1_func(0.0, epsilon_reg_head)
            d1_val_self = D1_func(0.0, epsilon_reg_head)
            
            sum_u_terms_j[0] += g0_head[0] * h1_val
            sum_u_terms_j[1] += g0_head[1] * h1_val
            sum_u_terms_j[2] += g0_head[2] * h1_val

            sum_w_terms_j[0] += 0.25 * m0_head[0] * d1_val_self
            sum_w_terms_j[1] += 0.25 * m0_head[1] * d1_val_self
            sum_w_terms_j[2] += 0.25 * m0_head[2] * d1_val_self
        else:
            h1_val = H1_func(r_mag_jh, epsilon_reg_head)
            h2_val = H2_func(r_mag_jh, epsilon_reg_head)
            q_val = Q_func(r_mag_jh, epsilon_reg_head)
            d1_val = D1_func(r_mag_jh, epsilon_reg_head)
            d2_val = D2_func(r_mag_jh, epsilon_reg_head)

            dot_g0h_rvec = g0_head[0]*r_vec_jh[0] + g0_head[1]*r_vec_jh[1] + g0_head[2]*r_vec_jh[2]
            term_uS_0 = g0_head[0] * h1_val + dot_g0h_rvec * r_vec_jh[0] * h2_val
            term_uS_1 = g0_head[1] * h1_val + dot_g0h_rvec * r_vec_jh[1] * h2_val
            term_uS_2 = g0_head[2] * h1_val + dot_g0h_rvec * r_vec_jh[2] * h2_val
            
            cross_m0h_rvec_0 = m0_head[1]*r_vec_jh[2] - m0_head[2]*r_vec_jh[1]
            cross_m0h_rvec_1 = m0_head[2]*r_vec_jh[0] - m0_head[0]*r_vec_jh[2]
            cross_m0h_rvec_2 = m0_head[0]*r_vec_jh[1] - m0_head[1]*r_vec_jh[0]
            term_uR_for_u_0 = 0.5 * cross_m0h_rvec_0 * q_val
            term_uR_for_u_1 = 0.5 * cross_m0h_rvec_1 * q_val
            term_uR_for_u_2 = 0.5 * cross_m0h_rvec_2 * q_val

            sum_u_terms_j[0] += term_uS_0 + term_uR_for_u_0
            sum_u_terms_j[1] += term_uS_1 + term_uR_for_u_1
            sum_u_terms_j[2] += term_uS_2 + term_uR_for_u_2

            cross_g0h_rvec_0 = g0_head[1]*r_vec_jh[2] - g0_head[2]*r_vec_jh[1]
            cross_g0h_rvec_1 = g0_head[2]*r_vec_jh[0] - g0_head[0]*r_vec_jh[2]
            cross_g0h_rvec_2 = g0_head[0]*r_vec_jh[1] - g0_head[1]*r_vec_jh[0]
            term_uR_for_w_0 = 0.5 * cross_g0h_rvec_0 * q_val
            term_uR_for_w_1 = 0.5 * cross_g0h_rvec_1 * q_val
            term_uR_for_w_2 = 0.5 * cross_g0h_rvec_2 * q_val
            
            dot_m0h_rvec = m0_head[0]*r_vec_jh[0] + m0_head[1]*r_vec_jh[1] + m0_head[2]*r_vec_jh[2]
            term_uD_0 = 0.25 * m0_head[0] * d1_val + 0.25 * dot_m0h_rvec * r_vec_jh[0] * d2_val
            term_uD_1 = 0.25 * m0_head[1] * d1_val + 0.25 * dot_m0h_rvec * r_vec_jh[1] * d2_val
            term_uD_2 = 0.25 * m0_head[2] * d1_val + 0.25 * dot_m0h_rvec * r_vec_jh[2] * d2_val

            sum_w_terms_j[0] += term_uR_for_w_0 + term_uD_0
            sum_w_terms_j[1] += term_uR_for_w_1 + term_uD_1
            sum_w_terms_j[2] += term_uR_for_w_2 + term_uD_2
        
        u_rod_out[j,0] = sum_u_terms_j[0] / mu_rod
        u_rod_out[j,1] = sum_u_terms_j[1] / mu_rod
        u_rod_out[j,2] = sum_u_terms_j[2] / mu_rod

        w_rod_out[j,0] = sum_w_terms_j[0] / mu_rod
        w_rod_out[j,1] = sum_w_terms_j[1] / mu_rod
        w_rod_out[j,2] = sum_w_terms_j[2] / mu_rod

    # Compute velocities at head center
    sum_u_head = np.zeros(3)
    sum_w_head = np.zeros(3)

    # Contribution from rod points to head
    for k in range(M_rod):
        Xk = X_rod[k]
        g0k = g0_all_rod[k]
        m0k = m0_all_rod[k]
        
        r_vec_hk = np.empty(3)
        r_vec_hk[0] = X_head[0] - Xk[0]
        r_vec_hk[1] = X_head[1] - Xk[1]
        r_vec_hk[2] = X_head[2] - Xk[2]
        
        r_mag_hk = np.sqrt(r_vec_hk[0]**2 + r_vec_hk[1]**2 + r_vec_hk[2]**2)

        if r_mag_hk < 1e-9:
            h1_val = H1_func(0.0, epsilon_reg_rod)
            d1_val_self = D1_func(0.0, epsilon_reg_rod)
            
            sum_u_head[0] += g0k[0] * h1_val
            sum_u_head[1] += g0k[1] * h1_val
            sum_u_head[2] += g0k[2] * h1_val

            sum_w_head[0] += 0.25 * m0k[0] * d1_val_self
            sum_w_head[1] += 0.25 * m0k[1] * d1_val_self
            sum_w_head[2] += 0.25 * m0k[2] * d1_val_self
        else:
            h1_val = H1_func(r_mag_hk, epsilon_reg_rod)
            h2_val = H2_func(r_mag_hk, epsilon_reg_rod)
            q_val = Q_func(r_mag_hk, epsilon_reg_rod)
            d1_val = D1_func(r_mag_hk, epsilon_reg_rod)
            d2_val = D2_func(r_mag_hk, epsilon_reg_rod)

            dot_g0k_rvec = g0k[0]*r_vec_hk[0] + g0k[1]*r_vec_hk[1] + g0k[2]*r_vec_hk[2]
            term_uS_0 = g0k[0] * h1_val + dot_g0k_rvec * r_vec_hk[0] * h2_val
            term_uS_1 = g0k[1] * h1_val + dot_g0k_rvec * r_vec_hk[1] * h2_val
            term_uS_2 = g0k[2] * h1_val + dot_g0k_rvec * r_vec_hk[2] * h2_val
            
            cross_m0k_rvec_0 = m0k[1]*r_vec_hk[2] - m0k[2]*r_vec_hk[1]
            cross_m0k_rvec_1 = m0k[2]*r_vec_hk[0] - m0k[0]*r_vec_hk[2]
            cross_m0k_rvec_2 = m0k[0]*r_vec_hk[1] - m0k[1]*r_vec_hk[0]
            term_uR_for_u_0 = 0.5 * cross_m0k_rvec_0 * q_val
            term_uR_for_u_1 = 0.5 * cross_m0k_rvec_1 * q_val
            term_uR_for_u_2 = 0.5 * cross_m0k_rvec_2 * q_val

            sum_u_head[0] += term_uS_0 + term_uR_for_u_0
            sum_u_head[1] += term_uS_1 + term_uR_for_u_1
            sum_u_head[2] += term_uS_2 + term_uR_for_u_2

            cross_g0k_rvec_0 = g0k[1]*r_vec_hk[2] - g0k[2]*r_vec_hk[1]
            cross_g0k_rvec_1 = g0k[2]*r_vec_hk[0] - g0k[0]*r_vec_hk[2]
            cross_g0k_rvec_2 = g0k[0]*r_vec_hk[1] - g0k[1]*r_vec_hk[0]
            term_uR_for_w_0 = 0.5 * cross_g0k_rvec_0 * q_val
            term_uR_for_w_1 = 0.5 * cross_g0k_rvec_1 * q_val
            term_uR_for_w_2 = 0.5 * cross_g0k_rvec_2 * q_val
            
            dot_m0k_rvec = m0k[0]*r_vec_hk[0] + m0k[1]*r_vec_hk[1] + m0k[2]*r_vec_hk[2]
            term_uD_0 = 0.25 * m0k[0] * d1_val + 0.25 * dot_m0k_rvec * r_vec_hk[0] * d2_val
            term_uD_1 = 0.25 * m0k[1] * d1_val + 0.25 * dot_m0k_rvec * r_vec_hk[1] * d2_val
            term_uD_2 = 0.25 * m0k[2] * d1_val + 0.25 * dot_m0k_rvec * r_vec_hk[2] * d2_val

            sum_w_head[0] += term_uR_for_w_0 + term_uD_0
            sum_w_head[1] += term_uR_for_w_1 + term_uD_1
            sum_w_head[2] += term_uR_for_w_2 + term_uD_2

    u_head_out[0] = sum_u_head[0] / mu_rod
    u_head_out[1] = sum_u_head[1] / mu_rod
    u_head_out[2] = sum_u_head[2] / mu_rod

    w_head_out[0] = sum_w_head[0] / mu_rod
    w_head_out[1] = sum_w_head[1] / mu_rod
    w_head_out[2] = sum_w_head[2] / mu_rod
            
    return u_rod_out, w_rod_out, u_head_out, w_head_out

# --- Spherical Head Class ---
class SphericalHead:
    def __init__(self, params, initial_position=None):
        self.p = params
        self.radius = self.p["head_radius"]
        self.epsilon_reg = self.radius  # Use radius as regularization parameter
        self.dt = self.p["dt"]
        
        # Head state variables
        if initial_position is not None:
            self.X = initial_position.copy()
        else:
            self.X = np.array([0.0, 0.0, -self.radius])  # Default position
            
        # Head reference frame (similar to rod directors)
        self.D1 = np.array([1.0, 0.0, 0.0])  # Head director 1
        self.D2 = np.array([0.0, 1.0, 0.0])  # Head director 2  
        self.D3 = np.array([0.0, 0.0, 1.0])  # Head director 3
        
        # Attachment point offset (where flagellum connects)
        self.attachment_offset = self.p["head_attachment_offset"] * self.radius
        
    def get_attachment_point(self):
        """Get the position where the flagellum attaches to the head."""
        # Attachment point is offset along D3 direction
        return self.X + self.attachment_offset * self.D3
        
    def update_state(self, u_head, w_head):
        """Update head position and orientation."""
        # Update position
        self.X += u_head * self.dt
        
        # Update orientation using angular velocity
        w_mag = np.linalg.norm(w_head)
        if w_mag > 1e-9:
            axis_e = w_head / w_mag
            angle_theta = w_mag * self.dt
            R_matrix = get_rodrigues_rotation_matrix(axis_e, angle_theta)
            self.D1 = self.D1 @ R_matrix.T
            self.D2 = self.D2 @ R_matrix.T
            self.D3 = self.D3 @ R_matrix.T

# --- Enhanced Rod Class with Head ---
class KirchhoffRodWithHead:
    def __init__(self, params):
        self.p = params
        self.M = self.p["M"]
        self.ds = self.p["ds"]
        self.mu = self.p["mu"]
        self.epsilon_reg = self.p["epsilon_reg"]
        self.dt = self.p["dt"]
        self.L_eff = self.p["L_eff"]

        # Material moduli
        self.a = np.array([self.p["a1"], self.p["a2"], self.p["a3"]])
        self.b = np.array([self.p["b1"], self.p["b2"], self.p["b3"]])
        
        # Rod state variables
        self.X = np.zeros((self.M, 3))
        self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3))
        self.D3 = np.zeros((self.M, 3))

        self.time = 0.0
        self.s_vals = np.arange(self.M) * self.ds

        # Intrinsic curvature and twist
        omega_vec = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])
        self.Omega = np.tile(omega_vec, (self.M, 1))

        # Initialize head if included
        self.include_head = self.p.get("include_head", False)
        if self.include_head:
            # Position head at the beginning of the rod
            head_initial_pos = np.array([0.0, 0.0, -self.p["head_radius"]])
            self.head = SphericalHead(self.p, head_initial_pos)

        # Initialize rod shape
        initial_shape = self.p.get("initial_shape", "straight")
        if initial_shape == "straight":
            xi = self.p["xi_pert"]
            orient_axis = self.p.get("straight_rod_orientation_axis", 'z')
            
            if self.include_head:
                # Start rod from head attachment point
                attachment_point = self.head.get_attachment_point()
                if orient_axis == 'x':
                    self.X[:, 0] = attachment_point[0] + self.s_vals
                    self.X[:, 1] = attachment_point[1]
                    self.X[:, 2] = attachment_point[2]
                    self.D3[:, 0] = 1.0
                    self.D1[:, 1] = 1.0
                    self.D2[:, 2] = 1.0
                elif orient_axis == 'y':
                    self.X[:, 0] = attachment_point[0]
                    self.X[:, 1] = attachment_point[1] + self.s_vals
                    self.X[:, 2] = attachment_point[2]
                    self.D3[:, 1] = 1.0
                    self.D1[:, 2] = 1.0
                    self.D2[:, 0] = 1.0
                else:  # Default 'z'
                    self.X[:, 0] = attachment_point[0]
                    self.X[:, 1] = attachment_point[1]
                    self.X[:, 2] = attachment_point[2] + self.s_vals
                    self.D3[:, 2] = 1.0
                    self.D1[:, 0] = 1.0
                    self.D2[:, 1] = 1.0
            else:
                if orient_axis == 'x':
                    self.X[:, 0] = self.s_vals
                    self.D3[:, 0] = 1.0
                    self.D1[:, 1] = 1.0
                    self.D2[:, 2] = 1.0
                elif orient_axis == 'y':
                    self.X[:, 1] = self.s_vals
                    self.D3[:, 1] = 1.0
                    self.D1[:, 2] = 1.0
                    self.D2[:, 0] = 1.0
                else:  # Default 'z'
                    self.X[:, 2] = self.s_vals
                    self.D3[:, 2] = 1.0
                    self.D1[:, 0] = 1.0
                    self.D2[:, 1] = 1.0
            
            # Apply initial perturbation rotation
            pert_rot = get_rodrigues_rotation_matrix(self.D3[0], xi)
            self.D1 = self.D1 @ pert_rot.T
            self.D2 = self.D2 @ pert_rot.T
            
        # Normalize directors
        for i in range(self.M):
            d1, d2 = self.D1[i].copy(), self.D2[i].copy()
            norm_d1 = np.linalg.norm(d1)
            if norm_d1 < 1e-9:
                self.D1[i] = np.array([1.0, 0.0, 0.0])
            else:
                self.D1[i] = d1 / norm_d1
            d2_ortho = d2 - np.dot(d2, self.D1[i]) * self.D1[i]
            norm_d2_ortho = np.linalg.norm(d2_ortho)
            if norm_d2_ortho < 1e-9:
                temp_vec = np.array([0.0, 1.0, 0.0]) if np.abs(self.D1[i, 0]) > 0.9 else np.array([1.0, 0.0, 0.0])
                self.D2[i] = temp_vec - np.dot(temp_vec, self.D1[i]) * self.D1[i]
                self.D2[i] /= np.linalg.norm(self.D2[i])
            else:
                self.D2[i] = d2_ortho / norm_d2_ortho
            self.D3[i] = np.cross(self.D1[i], self.D2[i])

    def _get_D_matrix(self, k):
        """Returns the 3x3 director matrix D_k (directors as rows)."""
        return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def update_intrinsic_curvature(self):
        """Updates the intrinsic curvature and twist based on the simulation scenario."""
        if self.p["scenario"] == "flagellar_wave":
            k = self.p["k_wave"]
            b = self.p["b_amp"]
            sigma = self.p["sigma_freq"]
            self.Omega[:, 0] = -k**2 * b * np.sin(k * self.s_vals + sigma * self.time)
            self.Omega[:, 1] = 0.0
            self.Omega[:, 2] = 0.0

    def compute_internal_forces_and_moments(self):
        """Compute F_{k+1/2} and N_{k+1/2}."""
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
        self.D_half_matrices = []
        
        for k in range(self.M - 1):
            Dk_mat = self._get_D_matrix(k)
            Dkp1_mat = self._get_D_matrix(k + 1)
            Ak = Dkp1_mat @ Dk_mat.T
            try:
                sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            except Exception as e:
                raise e
            D_half_k_mat = sqrt_Ak @ Dk_mat
            self.D_half_matrices.append(D_half_k_mat)
            D1_h, D2_h, D3_h = D_half_k_mat[0], D_half_k_mat[1], D_half_k_mat[2]
            
            dX_ds = (self.X[k + 1] - self.X[k]) / self.ds
            F_coeffs = np.array([self.b[0] * (np.dot(D1_h, dX_ds)),
                                self.b[1] * (np.dot(D2_h, dX_ds)),
                                self.b[2] * (np.dot(D3_h, dX_ds) - 1.0)])
            F_half[k] = F_coeffs[0] * D1_h + F_coeffs[1] * D2_h + F_coeffs[2] * D3_h
            
            dD1ds = (self.D1[k + 1] - self.D1[k]) / self.ds
            dD2ds = (self.D2[k + 1] - self.D2[k]) / self.ds
            dD3ds = (self.D3[k + 1] - self.D3[k]) / self.ds
            N_coeffs = np.array([self.a[0] * (np.dot(dD2ds, D3_h) - self.Omega[k, 0]),
                                self.a[1] * (np.dot(dD3ds, D1_h) - self.Omega[k, 1]),
                                self.a[2] * (np.dot(dD1ds, D2_h) - self.Omega[k, 2])])
            N_half[k] = N_coeffs[0] * D1_h + N_coeffs[1] * D2_h + N_coeffs[2] * D3_h
        
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        """Compute force and torque densities on the rod."""
        f_on_rod = np.zeros((self.M, 3))
        n_on_rod = np.zeros((self.M, 3))
        
        # Boundary conditions
        F_b_start, F_b_end = np.zeros(3), np.zeros(3)
        N_b_start, N_b_end = np.zeros(3), np.zeros(3)
        
        # Handle constraint at head attachment if head is included
        if self.include_head:
            # First point of rod is constrained to move with head attachment point
            attachment_point = self.head.get_attachment_point()
            # Apply constraint force to keep first rod point at attachment
            # Use a softer constraint to avoid instability
            constraint_stiffness = 10.0  # Reduced stiffness for stability
            constraint_force = constraint_stiffness * (attachment_point - self.X[0])
            F_b_start = constraint_force
        
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

    def compute_velocities(self, f_on_rod, n_on_rod):
        """Compute velocities including head contribution."""
        g0_all = f_on_rod * self.ds
        m0_all = n_on_rod * self.ds
        
        if self.include_head:
            # Head force/torque (could be from external forces or constraints)
            # For now, assume head force/torque comes from constraint with first rod point
            g0_head = np.zeros(3)  # No external force on head for now
            m0_head = np.zeros(3)  # No external torque on head for now
            
            return _compute_velocities_core_with_head(self.M, self.X, g0_all, m0_all,
                                                    self.head.X, g0_head, m0_head,
                                                    self.epsilon_reg, self.head.epsilon_reg, self.mu)
        else:
            # Use original velocity computation without head
            from paste import _compute_velocities_core
            u_rod, w_rod = _compute_velocities_core(self.M, self.X, g0_all, m0_all,
                                                   self.epsilon_reg, self.mu)
            return u_rod, w_rod, np.zeros(3), np.zeros(3)

    def update_state(self, u_rod, w_rod, u_head, w_head):
        """Update rod and head states."""
        # Update rod
        self.X += u_rod * self.dt
        for k in range(self.M):
            wk = w_rod[k]
            wk_mag = np.linalg.norm(wk)
            if wk_mag > 1e-9:
                axis_e = wk / wk_mag
                angle_theta = wk_mag * self.dt
                R_matrix = get_rodrigues_rotation_matrix(axis_e, angle_theta)
                self.D1[k] = self.D1[k] @ R_matrix.T
                self.D2[k] = self.D2[k] @ R_matrix.T
                self.D3[k] = self.D3[k] @ R_matrix.T

        # Update head
        if self.include_head:
            self.head.update_state(u_head, w_head)

    def simulation_step(self):
        """Perform a single simulation step."""
        self.update_intrinsic_curvature()
        F_half, N_half = self.compute_internal_forces_and_moments()
        f_on_rod, n_on_rod = self.compute_fluid_forces_on_rod(F_half, N_half)
        u_rod, w_rod, u_head, w_head = self.compute_velocities(f_on_rod, n_on_rod)
        self.update_state(u_rod, w_rod, u_head, w_head)
        self.time += self.dt
        return u_rod, w_rod, f_on_rod, n_on_rod, u_head, w_head

# --- Flow Field Visualization ---
def compute_flow_field(rod, grid_points):
    """Compute velocity field at grid points due to rod and head."""
    velocities = np.zeros_like(grid_points)
    
    # Get current forces and torques on rod
    F_half, N_half = rod.compute_internal_forces_and_moments()
    f_on_rod, n_on_rod = rod.compute_fluid_forces_on_rod(F_half, N_half)
    g0_all = f_on_rod * rod.ds
    m0_all = n_on_rod * rod.ds
    
    # Compute velocity at each grid point
    for i, eval_point in enumerate(grid_points):
        velocity = np.zeros(3)
        
        # Contribution from rod points
        for k in range(rod.M):
            # Stokeslet contribution
            vel_contrib = stokeslet_velocity(eval_point, rod.X[k], g0_all[k], 
                                           rod.mu, rod.epsilon_reg)
            velocity += vel_contrib
            
            # Rotlet contribution (simplified - just the translational part)
            r_vec = eval_point - rod.X[k]
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 1e-12:
                q_val = Q_func(r_mag, rod.epsilon_reg)
                cross_m_r = np.cross(m0_all[k], r_vec)
                velocity += 0.5 * cross_m_r * q_val / rod.mu
        
        # Contribution from head
        if rod.include_head:
            # Head contribution (assuming zero force/torque for visualization)
            g0_head = np.zeros(3)
            vel_contrib = stokeslet_velocity(eval_point, rod.head.X, g0_head,
                                           rod.mu, rod.head.epsilon_reg)
            velocity += vel_contrib
        
        velocities[i] = velocity
    
    return velocities

def create_flow_grid(rod, params):
    """Create a grid of points for flow field visualization."""
    extent = params["flow_grid_extent"]
    n_points = params["flow_grid_size"]
    
    # Determine grid bounds based on rod position
    if rod.include_head:
        all_points = np.vstack([rod.X, rod.head.X.reshape(1, -1)])
    else:
        all_points = rod.X
    
    center = np.mean(all_points, axis=0)
    
    # Create 2D grid in x-y plane at z = center[2]
    x_range = np.linspace(center[0] - extent/2, center[0] + extent/2, n_points)
    y_range = np.linspace(center[1] - extent/2, center[1] + extent/2, n_points)
    
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = np.full_like(X_grid, center[2])
    
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])
    
    return grid_points, X_grid, Y_grid

# --- Main Simulation and Animation ---
if __name__ == '__main__':
    rod = KirchhoffRodWithHead(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    
    # Lists to store historical data
    history_X = []
    history_u = []
    history_w = []
    history_f = []
    history_n = []
    history_head_X = []
    history_head_u = []
    history_head_w = []
    history_flow_field = []

    print(f"Starting simulation for {PARAMS['total_time']}s, with dt={PARAMS['dt']}s ({num_steps} steps).")
    print(f"Rod initial shape: {PARAMS.get('initial_shape', 'straight')}")
    print(f"Effective rod length: {rod.L_eff:.4f} um")
    print(f"Regularization epsilon: {rod.epsilon_reg:.4f} um")
    if rod.include_head:
        print(f"Head radius: {rod.head.radius:.4f} um")
        print(f"Head epsilon: {rod.head.epsilon_reg:.4f} um")

    start_time = time.time()

    for step in range(num_steps):
        # Perform one simulation step
        if rod.include_head:
            u_rod, w_rod, f_on_rod, n_on_rod, u_head, w_head = rod.simulation_step()
        else:
            u_rod, w_rod, f_on_rod, n_on_rod = rod.simulation_step()
            u_head, w_head = np.zeros(3), np.zeros(3)
            
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            history_u.append(u_rod.copy())
            history_w.append(w_rod.copy())
            history_f.append(f_on_rod.copy())
            history_n.append(n_on_rod.copy())
            
            if rod.include_head:
                history_head_X.append(rod.head.X.copy())
                history_head_u.append(u_head.copy())
                history_head_w.append(w_head.copy())
            else:
                history_head_X.append(np.zeros(3))
                history_head_u.append(np.zeros(3))
                history_head_w.append(np.zeros(3))
            
            # Compute flow field for visualization
            if PARAMS["visualize_flow"]:
                grid_points, _, _ = create_flow_grid(rod, PARAMS)
                flow_velocities = compute_flow_field(rod, grid_points)
                history_flow_field.append(flow_velocities.copy())
            
            if step % (PARAMS["animation_steps_skip"] * 10) == 0:
                current_sim_time = time.time()
                elapsed_wall_time = current_sim_time - start_time
                max_coord = np.max(np.abs(rod.X))
                if rod.include_head:
                    max_coord = max(max_coord, np.max(np.abs(rod.head.X)))
                print(f"Step {step}/{num_steps}, Sim Time: {step*PARAMS['dt']:.2e} s, "
                      f"Wall Time: {elapsed_wall_time:.2f}s. Max coord: {max_coord:.2f}")
            
            # Stability checks
            if np.max(np.abs(rod.X)) > rod.L_eff * 20:
                print("Simulation unstable. Coordinates exploded.")
                break
            if np.isnan(rod.X).any() or np.isinf(rod.X).any():
                print("NaN/Inf detected in coordinates. Simulation unstable.")
                break
            if np.isnan(rod.D1).any() or np.isnan(rod.D2).any() or np.isnan(rod.D3).any():
                print("NaN detected in directors. Simulation unstable.")
                break

    end_time = time.time()
    print(f"Simulation finished. Total wall time: {end_time - start_time:.2f} seconds.")

    # Save history if requested
    if history_X and PARAMS.get("save_history", False):
        print("\nSaving simulation history to text files...")
        try:
            output_dir = "simulation_history"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            num_frames = len(history_X)
            M = PARAMS['M']

            def save_history(filename, data, header):
                path = os.path.join(output_dir, filename)
                if len(data[0].shape) == 2:  # Rod data (frames, M, 3)
                    reshaped_data = np.array(data).reshape(num_frames * M, 3)
                else:  # Head data (frames, 3)
                    reshaped_data = np.array(data)
                np.savetxt(path, reshaped_data, fmt='%.8e', header=header, comments='')
                print(f"   -> Saved: {path}")

            save_history('history_X.txt', history_X, 'x y z')
            save_history('history_u.txt', history_u, 'ux uy uz')
            save_history('history_w.txt', history_w, 'wx wy wz')
            save_history('history_f.txt', history_f, 'fx fy fz')
            save_history('history_n.txt', history_n, 'nx ny nz')
            
            if rod.include_head:
                save_history('history_head_X.txt', history_head_X, 'x y z')
                save_history('history_head_u.txt', history_head_u, 'ux uy uz')
                save_history('history_head_w.txt', history_head_w, 'wx wy wz')

            # Save parameters
            with open(os.path.join(output_dir, 'simulation_params.txt'), 'w') as f:
                f.write(f"num_frames = {num_frames}\n")
                f.write(f"M = {M}\n")
                f.write(f"dt_snapshot = {PARAMS['animation_steps_skip'] * PARAMS['dt']}\n")
                for key, value in PARAMS.items():
                    f.write(f"{key} = {value}\n")
            print(f"   -> Saved: {os.path.join(output_dir, 'simulation_params.txt')}")

            print("\nData saved successfully.")

        except Exception as e:
            print(f"An error occurred while saving history data: {e}")

    # Animation
    fig = plt.figure(figsize=(15, 8))
    
    if PARAMS["visualize_flow"] and history_flow_field:
        # Create subplot for rod/head and flow field
        ax1 = fig.add_subplot(121, projection='3d')  # Rod/head
        ax2 = fig.add_subplot(122)  # Flow field
    else:
        ax1 = fig.add_subplot(111, projection='3d')  # Rod/head only
        ax2 = None

    # Rod visualization
    line, = ax1.plot([], [], [], 'b-o', lw=2, markersize=3, label='Flagellum')
    head_point = None
    if rod.include_head:
        head_point, = ax1.plot([], [], [], 'ro', markersize=10, label='Head')

    # Set up 3D plot limits
    if history_X:
        all_coords = np.concatenate(history_X, axis=0)
        if rod.include_head:
            head_coords = np.array(history_head_X)
            all_coords = np.vstack([all_coords, head_coords])
        
        x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        z_min, z_max = np.min(all_coords[:, 2]), np.max(all_coords[:, 2])
        center_x, center_y, center_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        plot_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min, rod.L_eff * 0.5]) * 0.6
        
        if plot_range < 1e-6:
            plot_range = rod.L_eff if rod.L_eff > 0 else 1.0
            
        ax1.set_xlim([center_x - plot_range, center_x + plot_range])
        ax1.set_ylim([center_y - plot_range, center_y + plot_range])
        ax1.set_zlim([center_z - plot_range, center_z + plot_range])
    else:
        lim = rod.L_eff / 2 if rod.L_eff > 0 else 1
        ax1.set_xlim([-lim, lim])
        ax1.set_ylim([-lim, lim])
        ax1.set_zlim([-lim / 2, lim / 2])

    ax1.set_xlabel("X (um)")
    ax1.set_ylabel("Y (um)")
    ax1.set_zlabel("Z (um)")
    ax1.set_title("Kirchhoff Rod with Spherical Head")
    ax1.view_init(elev=20., azim=-35)
    ax1.legend()

    # Flow field visualization setup
    if ax2 is not None and history_flow_field:
        grid_points, X_grid, Y_grid = create_flow_grid(rod, PARAMS)
        flow_quiver = None
        flow_contour = None
        ax2.set_xlabel("X (um)")
        ax2.set_ylabel("Y (um)")
        ax2.set_title("Flow Field")
        ax2.set_aspect('equal')

    time_text = ax1.text2D(0.05, 0.95, '', transform=ax1.transAxes)

    def init_animation():
        line.set_data([], [])
        line.set_3d_properties([])
        if head_point:
            head_point.set_data([], [])
            head_point.set_3d_properties([])
        time_text.set_text('')
        
        if ax2:
            ax2.clear()
            ax2.set_xlabel("X (um)")
            ax2.set_ylabel("Y (um)")
            ax2.set_title("Flow Field")
            ax2.set_aspect('equal')
        
        return line, head_point, time_text

    def update_animation(frame_idx):
        # Update rod
        X_data = history_X[frame_idx]
        line.set_data(X_data[:, 0], X_data[:, 1])
        line.set_3d_properties(X_data[:, 2])
        
        # Update head
        if head_point and rod.include_head:
            head_data = history_head_X[frame_idx]
            head_point.set_data([head_data[0]], [head_data[1]])
            head_point.set_3d_properties([head_data[2]])
        
        # Update time
        current_time = frame_idx * PARAMS["animation_steps_skip"] * PARAMS["dt"]
        time_text.set_text(f'Time: {current_time:.3e} s')
        
        # Update flow field
        if ax2 is not None and history_flow_field:
            ax2.clear()
            ax2.set_xlabel("X (um)")
            ax2.set_ylabel("Y (um)")
            ax2.set_title("Flow Field")
            ax2.set_aspect('equal')
            
            # Get flow data for this frame
            flow_velocities = history_flow_field[frame_idx]
            grid_points, X_grid, Y_grid = create_flow_grid(rod, PARAMS)
            
            # Reshape velocities for plotting
            U = flow_velocities[:, 0].reshape(X_grid.shape)
            V = flow_velocities[:, 1].reshape(X_grid.shape)
            speed = np.sqrt(U**2 + V**2)
            
            # Plot velocity magnitude as contour
            if PARAMS["flow_color_velocity"]:
                contour = ax2.contourf(X_grid, Y_grid, speed, levels=20, cmap='viridis', alpha=0.6)
            
            # Plot velocity vectors
            skip = max(1, PARAMS["flow_grid_size"] // 10)  # Skip some arrows for clarity
            scale = PARAMS["flow_arrow_scale"]
            
            if PARAMS["flow_color_velocity"]:
                # Color arrows by speed
                colors = speed[::skip, ::skip].ravel()
                ax2.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip], 
                          U[::skip, ::skip], V[::skip, ::skip],
                          colors, scale=scale, scale_units='xy', angles='xy',
                          cmap='plasma', alpha=0.8)
            else:
                ax2.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip], 
                          U[::skip, ::skip], V[::skip, ::skip],
                          scale=scale, scale_units='xy', angles='xy',
                          color='red', alpha=0.8)
            
            # Overlay rod position (projection onto flow plane)
            X_proj = X_data[:, 0]
            Y_proj = X_data[:, 1]
            ax2.plot(X_proj, Y_proj, 'b-', linewidth=2, label='Flagellum')
            
            # Overlay head position if present
            if rod.include_head:
                head_data = history_head_X[frame_idx]
                ax2.plot(head_data[0], head_data[1], 'ro', markersize=8, label='Head')
            
            ax2.legend()
            
            # Set consistent axis limits
            extent = PARAMS["flow_grid_extent"]
            if history_X:
                center = np.mean(history_X[frame_idx], axis=0)
                ax2.set_xlim([center[0] - extent/2, center[0] + extent/2])
                ax2.set_ylim([center[1] - extent/2, center[1] + extent/2])
        
        return line, head_point, time_text

    # Create animation
    if history_X:
        ani = FuncAnimation(fig, update_animation, frames=len(history_X),
                          init_func=init_animation, blit=False, interval=PARAMS["animation_interval"])
        
        # Save animation
        try:
            save_filename = 'enhanced_kirchhoff_rod_animation.mp4'
            print(f"Attempting to save animation as {save_filename}...")
            ani.save(save_filename, writer='ffmpeg', fps=15, dpi=150)
            print(f"Animation saved as {save_filename}")
        except Exception as e:
            print(f"Error saving as MP4: {e}. Make sure ffmpeg is installed and in PATH.")
        
        try:
            print("Attempting to save animation as GIF...")
            ani.save('enhanced_kirchhoff_rod_animation.gif', writer='pillow', fps=10, dpi=100)
            print("Animation saved as enhanced_kirchhoff_rod_animation.gif")
        except Exception as e:
            print(f"Error saving as GIF: {e}")
            print("Make sure a suitable writer (like pillow or imagemagick) is available.")

        plt.tight_layout()
        plt.show()
    else:
        print("No history data to animate.")

    # Additional analysis and plots
    if history_X and PARAMS["debugging"]:
        print("\nGenerating analysis plots...")
        
        # Plot head trajectory if included
        if rod.include_head and history_head_X:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            head_positions = np.array(history_head_X)
            time_points = np.arange(len(head_positions)) * PARAMS["animation_steps_skip"] * PARAMS["dt"]
            
            # Head position vs time
            axes[0, 0].plot(time_points, head_positions[:, 0], label='X')
            axes[0, 0].plot(time_points, head_positions[:, 1], label='Y')
            axes[0, 0].plot(time_points, head_positions[:, 2], label='Z')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Head Position (um)')
            axes[0, 0].set_title('Head Position vs Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Head velocity vs time
            if history_head_u:
                head_velocities = np.array(history_head_u)
                axes[0, 1].plot(time_points, head_velocities[:, 0], label='Vx')
                axes[0, 1].plot(time_points, head_velocities[:, 1], label='Vy')
                axes[0, 1].plot(time_points, head_velocities[:, 2], label='Vz')
                axes[0, 1].set_xlabel('Time (s)')
                axes[0, 1].set_ylabel('Head Velocity (um/s)')
                axes[0, 1].set_title('Head Velocity vs Time')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Head trajectory in 3D
            ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
            ax_3d.plot(head_positions[:, 0], head_positions[:, 1], head_positions[:, 2], 'r-')
            ax_3d.scatter(head_positions[0, 0], head_positions[0, 1], head_positions[0, 2], 
                         color='green', s=50, label='Start')
            ax_3d.scatter(head_positions[-1, 0], head_positions[-1, 1], head_positions[-1, 2], 
                         color='red', s=50, label='End')
            ax_3d.set_xlabel('X (um)')
            ax_3d.set_ylabel('Y (um)')
            ax_3d.set_zlabel('Z (um)')
            ax_3d.set_title('Head Trajectory')
            ax_3d.legend()
            
            # Swimming speed analysis
            if len(head_positions) > 1:
                swimming_speeds = []
                for i in range(1, len(head_positions)):
                    displacement = np.linalg.norm(head_positions[i] - head_positions[i-1])
                    dt_frame = PARAMS["animation_steps_skip"] * PARAMS["dt"]
                    speed = displacement / dt_frame
                    swimming_speeds.append(speed)
                
                axes[1, 1].plot(time_points[1:], swimming_speeds)
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Swimming Speed (um/s)')
                axes[1, 1].set_title('Instantaneous Swimming Speed')
                axes[1, 1].grid(True)
                
                avg_speed = np.mean(swimming_speeds)
                print(f"Average swimming speed: {avg_speed:.4f} um/s")
            
            plt.tight_layout()
            plt.show()
        
        # Flow field analysis
        if PARAMS["visualize_flow"] and history_flow_field:
            print("Analyzing flow field statistics...")
            
            # Compute statistics over time
            max_speeds = []
            avg_speeds = []
            
            for frame_idx, flow_velocities in enumerate(history_flow_field):
                speeds = np.linalg.norm(flow_velocities, axis=1)
                max_speeds.append(np.max(speeds))
                avg_speeds.append(np.mean(speeds))
            
            time_points = np.arange(len(max_speeds)) * PARAMS["animation_steps_skip"] * PARAMS["dt"]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(time_points, max_speeds, label='Max Speed')
            ax1.plot(time_points, avg_speeds, label='Avg Speed')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Flow Speed (um/s)')
            ax1.set_title('Flow Field Statistics vs Time')
            ax1.legend()
            ax1.grid(True)
            
            # Flow speed histogram for final frame
            if history_flow_field:
                final_speeds = np.linalg.norm(history_flow_field[-1], axis=1)
                ax2.hist(final_speeds, bins=30, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Flow Speed (um/s)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Flow Speed Distribution (Final Frame)')
                ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print(f"Max flow speed observed: {np.max(max_speeds):.4f} um/s")
            print(f"Average flow speed: {np.mean(avg_speeds):.4f} um/s")

    print("\nSimulation complete!")
    if rod.include_head:
        print("Enhanced features included:")
        print("- Spherical head with regularized Stokeslet")
        print("- Head-flagellum coupling")
        if PARAMS["visualize_flow"]:
            print("- Flow field visualization")
    else:
        print("Standard Kirchhoff rod simulation (no head)")
        if PARAMS["visualize_flow"]:
            print("- Flow field visualization")