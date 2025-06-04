import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba # Added Numba import

# --- Parameters ---
# (PARAMS dictionary remains the same as your last version)
PARAMS = {
    # Rod properties
    "M": 76,  # Number of immersed boundary points
    "ds": 0.0785,  # Meshwidth for rod (um)

    # Time integration
    "dt": 1.0e-6,  # Time step (s)
    "total_time": 0.01, # Total simulation time (s) # For circular, might need longer, e.g., 0.1s

    # Fluid properties
    "mu": 1.0e-6,  # Fluid viscosity (g um^-1 s^-1)

    # Regularization
    "epsilon_reg_factor": 6.0, # Factor for regularization parameter (epsilon_reg = epsilon_reg_factor * ds)

    # Material properties (g um^3 s^-2 for moduli)
    "a1": 3.5e-3,  # Bending modulus
    "a2": 3.5e-3,  # Bending modulus
    "a3": 3.5e-3,  # Twist modulus
    "b1": 8.0e-1,  # Shear modulus
    "b2": 8.0e-1,  # Shear modulus
    "b3": 8.0e-1,  # Stretch modulus

    # Intrinsic strain twist vector (um^-1)
    # Example for straight rod to helix (Fig. 2 top):
    # "Omega1": 1.3,
    # "Omega2": 0.0,
    # "Omega3": np.pi / 2.0,
    # Example for initially circular rod (based on Fig. 4, values might differ for open rod)
    "Omega1": 1.2, # Intrinsic curvature in D1 direction
    "Omega2": 0.0, # Intrinsic curvature in D2 direction
    "Omega3": 0.6, # Intrinsic twist

    # Initial shape configuration
    "initial_shape": "sinoidal", # Options: "straight", "circular", "sinoidal"

    # Parameters for "straight" initial_shape
    "xi_pert": 0.0001, # Perturbation for straight rod (from paper, section 6.1)

    # Parameters for "circular" initial_shape
    "r0_circ_val": None, # Specify a radius value directly, e.g., 2.5.
                         # If None, calculated to make a full circle from L_eff.
    "eta_pert": 0.00,  # Perturbation for circular rod (n in Eq. 47-49 of paper, section 6.2)

    # Animation settings
    "animation_interval": 50,  # ms between frames
    "animation_steps_skip": 200, # Number of simulation steps to skip per animation frame (Increased for faster animation with Numba)
}
PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]

# --- Helper Functions for Regularized Stokes Flow (Numba JITed) ---

@numba.njit(cache=True) # Added Numba JIT
def H1_func(r, epsilon_reg):
    # Eq. (63) / (A.19)
    return (2*epsilon_reg**2 + r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

@numba.njit(cache=True) # Added Numba JIT
def H2_func(r, epsilon_reg):
    # Eq. (64) / (A.20)
    return 1.0 / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

@numba.njit(cache=True) # Added Numba JIT
def Q_func(r, epsilon_reg):
    # Eq. (65) / (A.21)
    return (5*epsilon_reg**2 + 2*r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(5.0/2.0))

@numba.njit(cache=True) # Added Numba JIT
def D1_func(r, epsilon_reg):
    # Eq. (67) / (A.23)
    return (10*epsilon_reg**4 - 7*epsilon_reg**2*r**2 - 2*r**4) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))

@numba.njit(cache=True) # Added Numba JIT
def D2_func(r, epsilon_reg):
    # Eq. (68) / (A.24)
    return (21*epsilon_reg**2 + 6*r**2) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))

# --- Rotation Helpers (These remain standard Python as they use Scipy objects) ---
# (get_rotation_matrix_sqrt and get_rodrigues_rotation_matrix remain unchanged)
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
        if np.allclose(R_mat, np.eye(3)):
            return np.eye(3)
        try:
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
        except Exception as final_e:
            print(f"Critical Error in get_rotation_matrix_sqrt even after fallback. Matrix:\n{R_mat}\nError: {final_e}")
            raise final_e

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
        self.M = self.p["M"]
        self.ds = self.p["ds"]
        self.L_eff = (self.M - 1) * self.ds
        self.mu = self.p["mu"]
        self.epsilon_reg = self.p["epsilon_reg"]
        self.dt = self.p["dt"]

        self.a = np.array([self.p["a1"], self.p["a2"], self.p["a3"]])
        self.b = np.array([self.p["b1"], self.p["b2"], self.p["b3"]])
        self.Omega = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])

        self.X = np.zeros((self.M, 3))
        self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3))
        self.D3 = np.zeros((self.M, 3))
        
        s_vals = np.arange(self.M) * self.ds
        initial_shape = self.p.get("initial_shape", "straight")

        if initial_shape == "circular":
            if self.p.get("r0_circ_val") is not None:
                r0 = self.p["r0_circ_val"]
            elif self.L_eff > 0:
                r0 = self.L_eff / (2 * np.pi) 
            else:
                r0 = 1.0 
            
            eta = self.p.get("eta_pert", 0.0) 
            
            if r0 < 1e-9: 
                print("Warning: Circular radius r0 is very small. Defaulting to straight rod.")
                initial_shape = "straight" 
            else:
                theta_s_vals = s_vals / r0 
                self.X[:, 0] = r0 * np.cos(theta_s_vals)
                self.X[:, 1] = r0 * np.sin(theta_s_vals)
                self.X[:, 2] = 0.0
                self.D3[:, 0] = -np.sin(theta_s_vals)
                self.D3[:, 1] = np.cos(theta_s_vals)
                self.D3[:, 2] = 0.0
                z_unit_vec = np.array([0.0, 0.0, 1.0]) 
                alpha_s_vals = eta * np.sin(theta_s_vals) 
                cos_alpha_s = np.cos(alpha_s_vals)
                sin_alpha_s = np.sin(alpha_s_vals)
                for i in range(self.M):
                    r_s_i_vec = np.array([np.cos(theta_s_vals[i]), np.sin(theta_s_vals[i]), 0.0])
                    self.D1[i, :] = cos_alpha_s[i] * z_unit_vec + sin_alpha_s[i] * r_s_i_vec
                    self.D2[i, :] = -sin_alpha_s[i] * z_unit_vec + cos_alpha_s[i] * r_s_i_vec

        if initial_shape == "sinoidal":
            # Sinuoidal Rod Initialization 
            self.X[:, 0] = s_vals
            self.X[:, 1] = 2*np.sin(s_vals / 5)  # Example sine wave perturbation
            self.X[:, 2] = 2*np.cos(s_vals / 5)  # Example cosine wave perturbation

            self.D1[:, 0] = 1.0
            self.D1[:, 1] = 0.0
            self.D1[:, 2] = 0.0

            self.D2[:, 0] = 0.0
            self.D2[:, 1] = 1.0
            self.D2[:, 2] = 0.0

            self.D3[:, 0] = 0.0
            self.D3[:, 1] = 0.0
            self.D3[:, 2] = 1.0

        if initial_shape == "straight": # Handles default or fallback
            # --- Straight Rod Initialization (Eq. 40-43) ---
            xi = self.p["xi_pert"]
            self.X[:, 0] = 0.0
            self.X[:, 1] = 0.0
            self.X[:, 2] = (1 + xi) * s_vals # Small stretch/compression perturbation

            self.D1[:, 0] = 1.0
            self.D1[:, 1] = 0.0
            self.D1[:, 2] = 0.0

            # Constant rotation perturbation for D2, D3
            self.D2[:, 0] = 0.0
            self.D2[:, 1] = np.cos(xi)
            self.D2[:, 2] = -np.sin(xi)

            self.D3[:, 0] = 0.0
            self.D3[:, 1] = np.sin(xi)
            self.D3[:, 2] = np.cos(xi)

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
        return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def compute_internal_forces_and_moments(self):
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
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
            N_coeffs = np.array([ self.a[0] * (np.dot(dD2ds, D3_h) - self.Omega[0]),
                                  self.a[1] * (np.dot(dD3ds, D1_h) - self.Omega[1]),
                                  self.a[2] * (np.dot(dD1ds, D2_h) - self.Omega[2]) ])
            N_half[k] = N_coeffs[0]*D1_h + N_coeffs[1]*D2_h + N_coeffs[2]*D3_h
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        f_on_rod = np.zeros((self.M, 3)) 
        n_on_rod = np.zeros((self.M, 3)) 
        F_b_start, F_b_end, N_b_start, N_b_end = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for k in range(self.M): 
            F_prev = F_b_start if k == 0 else F_half[k-1]
            F_next = F_b_end if k == self.M - 1 else F_half[k]
            f_on_rod[k] = (F_next - F_prev) / self.ds
            N_prev = N_b_start if k == 0 else N_half[k-1]
            N_next = N_b_end if k == self.M - 1 else N_half[k]
            dNds_term = (N_next - N_prev) / self.ds
            cross_term = np.zeros(3)
            if k < self.M - 1: cross_term += np.cross((self.X[k+1]-self.X[k])/self.ds, F_next) 
            if k > 0: cross_term += np.cross((self.X[k]-self.X[k-1])/self.ds, F_prev) 
            n_on_rod[k] = dNds_term + 0.5 * cross_term
        return f_on_rod, n_on_rod

    def compute_velocities(self, f_on_rod, n_on_rod):
        """Steps 3 & 4: Compute u(X_j) and w(X_j). Wrapper for Numba core."""
        g0_all = f_on_rod * self.ds
        m0_all = n_on_rod * self.ds
        
        # Call the Numba-jitted core function
        u_rod, w_rod = _compute_velocities_core(
            self.M, self.X, g0_all, m0_all, self.epsilon_reg, self.mu
        )
        return u_rod, w_rod

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
        F_half, N_half = self.compute_internal_forces_and_moments()
        f_on_rod, n_on_rod = self.compute_fluid_forces_on_rod(F_half, N_half)
        u_rod, w_rod = self.compute_velocities(f_on_rod, n_on_rod)
        self.update_state(u_rod, w_rod)

# --- Main Simulation and Animation ---
if __name__ == '__main__':
    # Perform a "warm-up" call for Numba JIT compilation if desired
    # This compiles the functions on the first call.
    # For very short total_time, this compilation time can be noticeable.
    # For longer simulations, it's amortized.
    print("Warming up Numba JIT compilation (this might take a moment)...")
    try:
        # Create dummy data for a minimal rod to trigger compilation
        # Note: This warm-up needs to be careful not to pollute global state
        # or be too computationally expensive itself.
        # A simpler way is just to let the first simulation step do the JIT.
        # For this example, we'll let the first step handle it.
        # If you want a dedicated warm-up:
        # M_warmup = 2
        # X_warmup = np.zeros((M_warmup, 3))
        # g0_warmup = np.zeros((M_warmup, 3))
        # m0_warmup = np.zeros((M_warmup, 3))
        # eps_warmup = 0.1
        # mu_warmup = 1.0
        # _compute_velocities_core(M_warmup, X_warmup, g0_warmup, m0_warmup, eps_warmup, mu_warmup)
        # H1_func(0.1, eps_warmup) # etc. for other jitted functions
        print("Numba functions will be JIT compiled on their first use during the simulation.")
    except Exception as e:
        print(f"Numba warm-up or availability check issue: {e}")


    rod = KirchhoffRod(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    history_X = []

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

    import time # For timing
    start_time = time.time()

    for step in range(num_steps):
        rod.simulation_step()
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
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
    
    end_time = time.time()
    print(f"Simulation finished. Total wall time: {end_time - start_time:.2f} seconds.")

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

