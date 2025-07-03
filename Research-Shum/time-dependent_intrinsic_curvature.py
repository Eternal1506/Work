import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba # Added Numba import
import time # For timing
import os

# --- Parameters ---
# (PARAMS dictionary remains the same as your last version)
PARAMS = {
    # Scenario setup
    # Defines the simulation scenario. Options: "static_helix", "flagellar_wave".
    "scenario": "flagellar_wave", 

    # Rod properties
    "M": 600,  # Number of immersed boundary points
    "ds": 0.0667,  # Meshwidth for rod (um)

    # Time integration
    "dt": 1.0e-6,  # Time step (s)
    "total_time": 0.1, # Total simulation time (s) # For circular, might need longer, e.g., 0.1s

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
    "initial_shape": "straight", # Initial configuration of the rod. Options: "straight", "circular".
    "straight_rod_orientation_axis": 'z', # Axis along which a straight rod is oriented.
    "xi_pert": 0, # Perturbation angle for initial orientation (if applicable).

    # --- FLAGELLAR WAVE PARAMETERS (for 'flagellar_wave' scenario) ---
    # These parameters define a sinusoidal wave propagating along the flagellum.
    "k_wave": 2 * np.pi / 5.0, # Wave number (rad/um).
    "b_amp": 0.8,               # Wave amplitude (um).
    "sigma_freq": 275,           # Wave angular frequency (rad/s).

    # --- ANIMATION SETTINGS ---
    "animation_interval": 40, # Delay between frames in milliseconds for animation.
    "animation_steps_skip": 100, # Number of simulation steps to skip before saving a frame for animation.
    "debugging": False, # Enable/disable debugging plots.
    "debug_plot_interval_steps": 500, # How often to plot debug information (in simulation steps).
    "save_history": True, # Whether to save simulation data for later analysis.
}

# --- Scenario-specific overrides ---
# Adjusts parameters based on the chosen scenario.
if PARAMS["scenario"] == "static_helix":
    PARAMS.update({
        "Omega1": 1.3, "Omega3": np.pi / 2.0, # Non-zero intrinsic curvature (Omega1) and twist (Omega3) for a helix.
        "initial_shape": "straight", "straight_rod_orientation_axis": 'z'
    })

PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]

# --- Helper Functions for Regularized Stokes Flow ---

@numba.njit(cache=True)
def H1_func(r, epsilon_reg):
    # Eq. (63) / (A.19)
    return (2*epsilon_reg**2 + r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

@numba.njit(cache=True)  
def H2_func(r, epsilon_reg):
    # Eq. (64) / (A.20)
    return 1.0 / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

@numba.njit(cache=True) 
def Q_func(r, epsilon_reg):
    # Eq. (65) / (A.21)
    return (5*epsilon_reg**2 + 2*r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(5.0/2.0))

@numba.njit(cache=True)
def D1_func(r, epsilon_reg):
    # Eq. (67) / (A.23)
    return (10*epsilon_reg**4 - 7*epsilon_reg**2*r**2 - 2*r**4) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))

@numba.njit(cache=True)
def D2_func(r, epsilon_reg):
    # Eq. (68) / (A.24)
    return (21*epsilon_reg**2 + 6*r**2) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))

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
        # if np.allclose(R_mat, np.eye(3)):
        #     return np.eye(3)
        # try:
        #     U, _, Vh = np.linalg.svd(R_mat)
        #     R_mat_ortho = U @ Vh
        #     if np.linalg.det(R_mat_ortho) < 0: 
        #         Vh[-1,:] *= -1
        #         R_mat_ortho = U @ Vh
        #     rot = Rotation.from_matrix(R_mat_ortho)
        #     rotvec = rot.as_rotvec()
        #     sqrt_rotvec = rotvec * 0.5
        #     sqrt_R_mat = Rotation.from_rotvec(sqrt_rotvec).as_matrix()
        #     return sqrt_R_mat
        # except Exception as final_e:
        #     print(f"Critical Error in get_rotation_matrix_sqrt even after fallback. Matrix:\n{R_mat}\nError: {final_e}")
        #     raise final_e

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
def _compute_velocities_core(M_rod, X_rod, g0_all_rod, m0_all_rod, epsilon_reg_rod, mu_rod):
    """
    Numba-jitted core loop for velocity computation.
    All inputs must be Numba-compatible (e.g., NumPy arrays, scalars).
    """
    u_rod_out = np.zeros((M_rod, 3)) 
    w_rod_out = np.zeros((M_rod, 3)) 

    for j in range(M_rod): 
        Xj = X_rod[j]
        sum_u_terms_j = np.zeros(3)
        sum_w_terms_j = np.zeros(3)

        for k in range(M_rod): 
            Xk = X_rod[k]
            g0k = g0_all_rod[k] 
            m0k = m0_all_rod[k] 
            
            r_vec_jk = np.empty(3) # Pre-allocate for Numba
            r_vec_jk[0] = Xj[0] - Xk[0]
            r_vec_jk[1] = Xj[1] - Xk[1]
            r_vec_jk[2] = Xj[2] - Xk[2]
            
            r_mag_jk_sq = r_vec_jk[0]**2 + r_vec_jk[1]**2 + r_vec_jk[2]**2
            r_mag_jk = np.sqrt(r_mag_jk_sq)

            if r_mag_jk < 1e-9: 
                h1_val = H1_func(0.0, epsilon_reg_rod)
                d1_val_self = D1_func(0.0, epsilon_reg_rod)
                
                # sum_u_terms_j += g0k * h1_val
                sum_u_terms_j[0] += g0k[0] * h1_val
                sum_u_terms_j[1] += g0k[1] * h1_val
                sum_u_terms_j[2] += g0k[2] * h1_val

                # sum_w_terms_j += 0.25 * m0k * d1_val_self
                sum_w_terms_j[0] += 0.25 * m0k[0] * d1_val_self
                sum_w_terms_j[1] += 0.25 * m0k[1] * d1_val_self
                sum_w_terms_j[2] += 0.25 * m0k[2] * d1_val_self
            else:
                h1_val = H1_func(r_mag_jk, epsilon_reg_rod)
                h2_val = H2_func(r_mag_jk, epsilon_reg_rod)
                q_val = Q_func(r_mag_jk, epsilon_reg_rod)
                d1_val = D1_func(r_mag_jk, epsilon_reg_rod)
                d2_val = D2_func(r_mag_jk, epsilon_reg_rod)

                # Linear velocity contributions
                # Term from u_S[g0k]: (g0k H1) + (g0k . r_vec_jk) r_vec_jk H2
                dot_g0k_rvec = g0k[0]*r_vec_jk[0] + g0k[1]*r_vec_jk[1] + g0k[2]*r_vec_jk[2]
                term_uS_0 = g0k[0] * h1_val + dot_g0k_rvec * r_vec_jk[0] * h2_val
                term_uS_1 = g0k[1] * h1_val + dot_g0k_rvec * r_vec_jk[1] * h2_val
                term_uS_2 = g0k[2] * h1_val + dot_g0k_rvec * r_vec_jk[2] * h2_val
                
                # Term from u_R[m0k]: (1/2) (m0k x r_vec_jk) Q
                cross_m0k_rvec_0 = m0k[1]*r_vec_jk[2] - m0k[2]*r_vec_jk[1]
                cross_m0k_rvec_1 = m0k[2]*r_vec_jk[0] - m0k[0]*r_vec_jk[2]
                cross_m0k_rvec_2 = m0k[0]*r_vec_jk[1] - m0k[1]*r_vec_jk[0]
                term_uR_for_u_0 = 0.5 * cross_m0k_rvec_0 * q_val
                term_uR_for_u_1 = 0.5 * cross_m0k_rvec_1 * q_val
                term_uR_for_u_2 = 0.5 * cross_m0k_rvec_2 * q_val

                sum_u_terms_j[0] += term_uS_0 + term_uR_for_u_0
                sum_u_terms_j[1] += term_uS_1 + term_uR_for_u_1
                sum_u_terms_j[2] += term_uS_2 + term_uR_for_u_2

                # Angular velocity contributions
                # Term from u_R[g0k]: (1/2) (g0k x r_vec_jk) Q
                cross_g0k_rvec_0 = g0k[1]*r_vec_jk[2] - g0k[2]*r_vec_jk[1]
                cross_g0k_rvec_1 = g0k[2]*r_vec_jk[0] - g0k[0]*r_vec_jk[2]
                cross_g0k_rvec_2 = g0k[0]*r_vec_jk[1] - g0k[1]*r_vec_jk[0]
                term_uR_for_w_0 = 0.5 * cross_g0k_rvec_0 * q_val
                term_uR_for_w_1 = 0.5 * cross_g0k_rvec_1 * q_val
                term_uR_for_w_2 = 0.5 * cross_g0k_rvec_2 * q_val
                
                # Term from u_D[m0k]: (1/4) m0k D1 + (1/4) (m0k . r_vec_jk) r_vec_jk D2
                dot_m0k_rvec = m0k[0]*r_vec_jk[0] + m0k[1]*r_vec_jk[1] + m0k[2]*r_vec_jk[2]
                term_uD_0 = 0.25 * m0k[0] * d1_val + 0.25 * dot_m0k_rvec * r_vec_jk[0] * d2_val
                term_uD_1 = 0.25 * m0k[1] * d1_val + 0.25 * dot_m0k_rvec * r_vec_jk[1] * d2_val
                term_uD_2 = 0.25 * m0k[2] * d1_val + 0.25 * dot_m0k_rvec * r_vec_jk[2] * d2_val

                sum_w_terms_j[0] += term_uR_for_w_0 + term_uD_0
                sum_w_terms_j[1] += term_uR_for_w_1 + term_uD_1
                sum_w_terms_j[2] += term_uR_for_w_2 + term_uD_2
        
        u_rod_out[j,0] = sum_u_terms_j[0] / mu_rod
        u_rod_out[j,1] = sum_u_terms_j[1] / mu_rod
        u_rod_out[j,2] = sum_u_terms_j[2] / mu_rod

        w_rod_out[j,0] = sum_w_terms_j[0] / mu_rod
        w_rod_out[j,1] = sum_w_terms_j[1] / mu_rod
        w_rod_out[j,2] = sum_w_terms_j[2] / mu_rod
            
    return u_rod_out, w_rod_out


# --- Rod Class ---
class KirchhoffRod:
    def __init__(self, params):
        self.p = params
        self.M = self.p["M"] # Number of points.
        self.ds = self.p["ds"] # Meshwidth.
        self.mu = self.p["mu"] # Fluid viscosity.
        self.epsilon_reg = self.p["epsilon_reg"] # Regularization parameter.
        self.dt = self.p["dt"] # Time step.
        self.L_eff = self.p["L_eff"] # Effective length of the rod.

        # Material moduli for bending/twist (a) and shear/stretch (b).
        self.a = np.array([self.p["a1"],self.p["a2"],self.p["a3"]])
        self.b = np.array([self.p["b1"],self.p["b2"],self.p["b3"]])
        
        # Rod state variables:
        self.X = np.zeros((self.M, 3)) # Centerline coordinates of the rod.
        self.D1 = np.zeros((self.M, 3)) # Orthonormal director D1.
        self.D2 = np.zeros((self.M, 3)) # Orthonormal director D2.
        self.D3 = np.zeros((self.M, 3)) # Orthonormal director D3 (tangent-like).

        self.time = 0.0 # Current simulation time.
        self.s_vals = np.arange(self.M) * self.ds # Lagrangian parameter 's' values along the rod.

        # Intrinsic curvature and twist (Omega).
        # Tiled to have a value for each point on the rod.
        omega_vec = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])
        self.Omega = np.tile(omega_vec, (self.M, 1))

        # --- Initial Shape Setup ---
        initial_shape = self.p.get("initial_shape", "straight")

        if initial_shape == "straight":
            # Initializes the rod as a straight line along a specified axis.
            xi = self.p["xi_pert"] # Initial perturbation for the orientation.
            orient_axis = self.p.get("straight_rod_orientation_axis", 'z')
            if orient_axis == 'x':
                self.X[:, 0] = self.s_vals; self.D3[:, 0] = 1.0; self.D1[:, 1] = 1.0; self.D2[:, 2] = 1.0
            elif orient_axis == 'y':
                self.X[:, 1] = self.s_vals; self.D3[:, 1] = 1.0; self.D1[:, 2] = 1.0; self.D2[:, 0] = 1.0
            else: # Default 'z'
                self.X[:, 2] = self.s_vals; self.D3[:, 2] = 1.0; self.D1[:, 0] = 1.0; self.D2[:, 1] = 1.0
            
            # Apply initial perturbation rotation.
            pert_rot = get_rodrigues_rotation_matrix(self.D3[0], xi)
            self.D1 = self.D1 @ pert_rot.T; self.D2 = self.D2 @ pert_rot.T
            
        for i in range(self.M):
            d1, d2 = self.D1[i].copy(), self.D2[i].copy()
            norm_d1 = np.linalg.norm(d1)
            if norm_d1 < 1e-9: self.D1[i] = np.array([1.0,0.0,0.0])
            else: self.D1[i] = d1 / norm_d1
            d2_ortho = d2 - np.dot(d2, self.D1[i]) * self.D1[i]
            norm_d2_ortho = np.linalg.norm(d2_ortho)
            if norm_d2_ortho < 1e-9: 
                 temp_vec = np.array([0.0,1.0,0.0]) if np.abs(self.D1[i,0]) > 0.9 else np.array([1.0,0.0,0.0])
                 self.D2[i] = temp_vec - np.dot(temp_vec, self.D1[i]) * self.D1[i]
                 self.D2[i] /= np.linalg.norm(self.D2[i])
            else: self.D2[i] = d2_ortho / norm_d2_ortho
            self.D3[i] = np.cross(self.D1[i], self.D2[i])

    def _get_D_matrix(self, k):
        """Returns the 3x3 director matrix D_k (directors as rows)."""
        return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def update_intrinsic_curvature(self):
        """
        Updates the intrinsic curvature and twist (Omega) based on the simulation scenario.
        For "flagellar_wave" scenario, it sets up a sinusoidal propagating wave.
        """
        if self.p["scenario"] == "flagellar_wave":
            k = self.p["k_wave"]     # Wave number.
            b = self.p["b_amp"]      # Wave amplitude.
            sigma = self.p["sigma_freq"] # Wave angular frequency.
            # Sets Omega1 component to create the bending wave.
            self.Omega[:, 0] = -k**2 * b * np.sin(k * self.s_vals + sigma * self.time)
            self.Omega[:, 1] = 0.0
            self.Omega[:, 2] = 0.0

    def compute_internal_forces_and_moments(self):
        """Step 1: Compute F_{k+1/2} and N_{k+1/2}."""
        # These are forces/moments on M-1 segments
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
        # Store D_half_matrices for use in F_half, N_half component calculation
        self.D_half_matrices = [] 
        for k in range(self.M - 1): 
            Dk_mat = self._get_D_matrix(k)    
            Dkp1_mat = self._get_D_matrix(k+1) 
            Ak = Dkp1_mat @ Dk_mat.T 
            try: sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            except Exception as e: raise e
            D_half_k_mat = sqrt_Ak @ Dk_mat 
            self.D_half_matrices.append(D_half_k_mat)
            D1_h, D2_h, D3_h = D_half_k_mat[0], D_half_k_mat[1], D_half_k_mat[2]
            dX_ds = (self.X[k+1] - self.X[k]) / self.ds
            F_coeffs = np.array([ self.b[0] * (np.dot(D1_h, dX_ds)),
                                  self.b[1] * (np.dot(D2_h, dX_ds)),
                                  self.b[2] * (np.dot(D3_h, dX_ds) - 1.0) ])
            F_half[k] = F_coeffs[0]*D1_h + F_coeffs[1]*D2_h + F_coeffs[2]*D3_h
            dD1ds,dD2ds,dD3ds = (self.D1[k+1]-self.D1[k])/self.ds, (self.D2[k+1]-self.D2[k])/self.ds, (self.D3[k+1]-self.D3[k])/self.ds
            N_coeffs = np.array([ self.a[0] * (np.dot(dD2ds, D3_h) - self.Omega[k,0]),
                                  self.a[1] * (np.dot(dD3ds, D1_h) - self.Omega[k,1]),
                                  self.a[2] * (np.dot(dD1ds, D2_h) - self.Omega[k,2]) ])
            N_half[k] = N_coeffs[0]*D1_h + N_coeffs[1]*D2_h + N_coeffs[2]*D3_h
        return F_half, N_half

    def compute_fluid_forces_on_rod(self,F_half,N_half):
        """
        Computes the force density (f_on_rod) and torque density (n_on_rod) exerted by the fluid on the rod.
        These are derived from the force and torque balance equations.
        Specifically, it implements the discrete versions of Eq. (3a) and Eq. (3b).
        """
        f_on_rod=np.zeros((self.M,3));n_on_rod=np.zeros((self.M,3))
        # Boundary conditions (zero force/torque at ends assumed for this implementation).
        F_b_start,F_b_end,N_b_start,N_b_end=np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
        for k in range(self.M): # Iterate through each point on the rod.
            # Central difference approximation for spatial derivatives.
            F_prev=F_b_start if k==0 else F_half[k-1]
            F_next=F_b_end if k==self.M-1 else F_half[k]
            f_on_rod[k]=(F_next-F_prev)/self.ds # Force density: derivative of internal force (Eq. 3a).

            N_prev=N_b_start if k==0 else N_half[k-1]
            N_next=N_b_end if k==self.M-1 else N_half[k]
            dN_ds=(N_next-N_prev)/self.ds # Derivative of internal moment.

            cross_term=np.zeros(3)
            # Cross product term from torque balance equation (Eq. 3b).
            if k<self.M-1: cross_term+=np.cross((self.X[k+1]-self.X[k])/self.ds,F_next)
            if k>0: cross_term+=np.cross((self.X[k]-self.X[k-1])/self.ds,F_prev)
            n_on_rod[k]=dN_ds+.5*cross_term # Torque density (Eq. 3b).
            
        return f_on_rod,n_on_rod

    def compute_velocities(self,f_on_rod,n_on_rod):
        """
        Computes the linear (u_rod) and angular (w_rod) velocities of the rod
        by calling the Numba-optimized core function.
        This step couples the rod dynamics to the fluid dynamics using the regularized Stokes formulation.
        """
        # Convert force/torque densities to point forces/torques by multiplying by ds.
        g0_all=f_on_rod*self.ds 
        m0_all=n_on_rod*self.ds
        return _compute_velocities_core(self.M,self.X,g0_all,m0_all,self.epsilon_reg,self.mu)

    def update_state(self, u_rod, w_rod):
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

    def simulation_step(self):
        """
        Performs a single simulation step for the Kirchhoff rod.
        This involves:
        1. Updating intrinsic curvature/twist (if time-dependent).
        2. Computing internal elastic forces and moments.
        3. Computing fluid forces and torques acting on the rod.
        4. Calculating the resulting linear and angular velocities from fluid interaction.
        5. Updating the rod's position and orientation.
        """
        self.update_intrinsic_curvature()
        F_half,N_half=self.compute_internal_forces_and_moments()
        f_on_rod,n_on_rod=self.compute_fluid_forces_on_rod(F_half,N_half)
        u_rod,w_rod=self.compute_velocities(f_on_rod,n_on_rod)
        self.update_state(u_rod,w_rod)
        self.time += self.dt # Advance simulation time.
        return u_rod, w_rod, f_on_rod, n_on_rod # Return for debugging/analysis.

# --- Main Simulation and Animation ---
if __name__ == '__main__':
    rod = KirchhoffRod(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    # Lists to store historical data for animation and saving.
    history_X = [] # Rod centerline positions.
    history_u = [] # Linear velocities.
    history_w = [] # Angular velocities.
    history_f = [] # Fluid forces.
    history_n = [] # Fluid torques.

    print(f"Starting simulation for {PARAMS['total_time']}s, with dt={PARAMS['dt']}s ({num_steps} steps).")
    print(f"Rod initial shape: {PARAMS.get('initial_shape', 'straight')}")
    print(f"Effective rod length: {rod.L_eff:.4f} um")
    print(f"Regularization epsilon: {rod.epsilon_reg:.4f} um")
    if PARAMS.get('initial_shape') == 'circular':
        if rod.L_eff > 0 and PARAMS.get("r0_circ_val") is None:
             r0_calc = rod.L_eff / (2 * np.pi)
             print(f"Calculated circular radius r0: {r0_calc:.4f} um (for a full circle)")
        elif PARAMS.get("r0_circ_val") is not None:
             print(f"Specified circular radius r0: {PARAMS['r0_circ_val']:.4f} um")

    start_time = time.time()

    for step in range(num_steps):
        # Perform one simulation step.
        u_rod, w_rod, f_on_rod, n_on_rod = rod.simulation_step()
            
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            history_u.append(u_rod.copy())
            history_w.append(w_rod.copy())
            history_f.append(f_on_rod.copy())
            history_n.append(n_on_rod.copy())
            if step % (PARAMS["animation_steps_skip"]*10) == 0 : 
                current_sim_time = time.time()
                elapsed_wall_time = current_sim_time - start_time
                print(f"Step {step}/{num_steps}, Sim Time: {step*PARAMS['dt']:.2e} s, Wall Time: {elapsed_wall_time:.2f}s. Max X: {np.max(np.abs(rod.X)):.2f}")
            
            if np.max(np.abs(rod.X)) > rod.L_eff * 20 : 
                 print("Simulation unstable. Coordinates exploded.")
                 break
            if np.isnan(rod.X).any() or np.isinf(rod.X).any():
                 print("NaN/Inf detected in coordinates. Simulation unstable.")
                 break
            if np.isnan(rod.D1).any() or np.isnan(rod.D2).any() or np.isnan(rod.D3).any():
                 print("NaN detected in directors. Simulation unstable.")
                 break

        # Debugging plots for velocities, forces, and torques at a specific interval.
        if step % PARAMS["debug_plot_interval_steps"] == 0 and PARAMS["debugging"]:
            print(f"Plotting debug information at simulation time: {rod.time:.4f}s")
            
            # Plot Intrinsic Curvature Profile (Omega1).
            plt.figure(figsize=(7,4))
            plt.plot(rod.s_vals, rod.Omega[:, 0], 'o-')
            plt.xlabel("s (um)")
            plt.ylabel(r"$\Omega_1(s)$")
            plt.title(f"Intrinsic Curvature Profile at t = {rod.time:.4f} s")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot Linear and Angular Velocities.
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(rod.s_vals, u_rod[:, 0], label='$u_x$')
            plt.plot(rod.s_vals, u_rod[:, 1], label='$u_y$')
            plt.plot(rod.s_vals, u_rod[:, 2], label='$u_z$')
            plt.xlabel("s (um)")
            plt.ylabel("Linear Velocity (um/s)")
            plt.title(f"Linear Velocities at t = {rod.time:.4f} s")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(rod.s_vals, w_rod[:, 0], label=r'$\omega_x$')
            plt.plot(rod.s_vals, w_rod[:, 1], label=r'$\omega_y$')
            plt.plot(rod.s_vals, w_rod[:, 2], label=r'$\omega_z$')
            plt.xlabel("s (um)")
            plt.ylabel("Angular Velocity (rad/s)")
            plt.title(f"Angular Velocities at t = {rod.time:.4f} s")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot Fluid Forces and Torques.
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(rod.s_vals, f_on_rod[:, 0], label='$f_x$')
            plt.plot(rod.s_vals, f_on_rod[:, 1], label='$f_y$')
            plt.plot(rod.s_vals, f_on_rod[:, 2], label='$f_z$')
            plt.xlabel("s (um)")
            plt.ylabel("Force (g um^-1 s^-2)")
            plt.title(f"Fluid Forces on Rod at t = {rod.time:.4f} s")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(rod.s_vals, n_on_rod[:, 0], label='$n_x$')
            plt.plot(rod.s_vals, n_on_rod[:, 1], label='$n_y$')
            plt.plot(rod.s_vals, n_on_rod[:, 2], label='$n_z$')
            plt.xlabel("s (um)")
            plt.ylabel("Torque (g um s^-2)")
            plt.title(f"Fluid Torques on Rod at t = {rod.time:.4f} s")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    end_time = time.time()
    print(f"Simulation finished. Total wall time: {end_time - start_time:.2f} seconds.")

    if history_X and PARAMS.get("save_history", False): # Check if simulation ran and produced history.
        print("\n Saving simulation history to text files...")
        try:
            output_dir = "simulation_history"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            num_frames = len(history_X)
            M = PARAMS['M']

            # Helper function to save data to a text file.
            def save_history(filename, data, header):
                path = os.path.join(output_dir, filename)
                # Reshape data to (num_frames * M, 3) for easier saving.
                reshaped_data = np.array(data).reshape(num_frames * M, 3) 
                np.savetxt(path, reshaped_data, fmt='%.8e', header=header, comments='')
                print(f"   -> Saved: {path}")

            # Save all recorded simulation histories.
            save_history('history_X.txt', history_X, 'x y z')
            save_history('history_u.txt', history_u, 'ux uy uz')
            save_history('history_w.txt', history_w, 'wx wy wz')
            save_history('history_f.txt', history_f, 'fx fy fz')
            save_history('history_n.txt', history_n, 'nx ny nz')

            # Save simulation parameters for reproducibility.
            with open(os.path.join(output_dir, 'simulation_params.txt'), 'w') as f:
                f.write(f"num_frames = {num_frames}\n")
                f.write(f"M = {M}\n")
                f.write(f"dt_snapshot = {PARAMS['animation_steps_skip'] * PARAMS['dt']}\n")
                for key, value in PARAMS.items():
                    f.write(f"{key} = {value}\n")
            print(f"   -> Saved: {os.path.join(output_dir, 'simulation_params.txt')}")


            print("\nData saved successfully.")
            print("To load data back into a 3D numpy array (frames, M, 3), for example:")
            print(f"loaded_X = np.loadtxt('{os.path.join(output_dir, 'history_X.txt')}', skiprows=1).reshape({num_frames}, {M}, 3)")

        except Exception as e:
            print(f"An error occurred while saving history data: {e}")


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=3)

    if history_X:
        all_coords = np.concatenate(history_X, axis=0)
        x_min, x_max = np.min(all_coords[:,0]), np.max(all_coords[:,0])
        y_min, y_max = np.min(all_coords[:,1]), np.max(all_coords[:,1])
        z_min, z_max = np.min(all_coords[:,2]), np.max(all_coords[:,2])
        center_x,center_y,center_z = (x_min+x_max)/2,(y_min+y_max)/2,(z_min+z_max)/2
        plot_range = np.max([x_max-x_min, y_max-y_min, z_max-z_min, rod.L_eff*0.5])*0.6 
        if plot_range < 1e-6 : plot_range = rod.L_eff if rod.L_eff > 0 else 1.0
        ax.set_xlim([center_x-plot_range, center_x+plot_range])
        ax.set_ylim([center_y-plot_range, center_y+plot_range])
        ax.set_zlim([center_z-plot_range, center_z+plot_range])
        try: ax.set_aspect('equal', adjustable='box')
        except NotImplementedError: print("Warning: ax.set_aspect('equal') not fully supported for 3D plots by this matplotlib backend.")
    else: 
        lim = rod.L_eff/2 if rod.L_eff > 0 else 1
        ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim/2, lim/2])

    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)"); ax.set_zlabel("Z (um)")
    ax.set_title(f"Kirchhoff Rod Dynamics (Initial: {PARAMS.get('initial_shape', 'straight')})")
    ax.view_init(elev=20., azim=-35) 
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def init_animation():
        line.set_data([], []); line.set_3d_properties([])
        time_text.set_text('')
        return line, time_text

    def update_animation(frame_idx):
        X_data = history_X[frame_idx]
        line.set_data(X_data[:,0], X_data[:,1]); line.set_3d_properties(X_data[:,2])
        current_time = frame_idx * PARAMS["animation_steps_skip"] * PARAMS["dt"]
        time_text.set_text(f'Time: {current_time:.3e} s')
        return line, time_text

    if history_X:
        ani = FuncAnimation(fig, update_animation, frames=len(history_X),
                            init_func=init_animation, blit=False, interval=PARAMS["animation_interval"])
        try:
            save_filename = 'kirchhoff_rod_animation.mp4'
            print(f"Attempting to save animation as {save_filename}...")
            ani.save(save_filename, writer='ffmpeg', fps=15, dpi=150)
            print(f"Animation saved as {save_filename}")
        except Exception as e:
            print(f"Error saving as MP4: {e}. Make sure ffmpeg is installed and in PATH.")
        
        try:
            print("Attempting to save animation as GIF...")
            # Using pillow writer (often built-in)
            ani.save('kirchhoff_rod_animation.gif', writer='pillow', fps=10, dpi=100)
            # Or using imagemagick (if pillow fails or for better quality)
            # ani.save('kirchhoff_rod_animation.gif', writer='imagemagick', fps=10, dpi=100)
            print("Animation saved as kirchhoff_rod_animation.gif")
        except Exception as e:
            print(f"Error saving as GIF: {e}")
            print("Make sure a suitable writer (like pillow or imagemagick) is available.")

        plt.tight_layout(); plt.show()
    else: print("No history data to animate.")

