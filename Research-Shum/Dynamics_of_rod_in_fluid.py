import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

# --- Parameters ---
# These parameters are based on Tables 1 & 2 and Section 6.1 (open rod)
# or Section 6.2 (closed rod initial shape) of the paper.

PARAMS = {
    # Rod properties
    "M": 76,  # Number of immersed boundary points
    "ds": 0.0785,  # Meshwidth for rod (um)

    # Time integration
    "dt": 1.0e-6,  # Time step (s)
    "total_time": 0.005, # Total simulation time (s) # For circular, might need longer, e.g., 0.1s

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
    # "Omega1": 1.2, # Intrinsic curvature in D1 direction
    # "Omega2": 0.0, # Intrinsic curvature in D2 direction
    # "Omega3": 0.6, # Intrinsic twist

    "Omega1": 0.0, # Intrinsic curvature in D1 direction
    "Omega2": 0.0, # Intrinsic curvature in D2 direction
    "Omega3": 0.0, # Intrinsic twist
    # Note: Omega values can be adjusted based on the initial shape and desired dynamics.

    # Initial shape configuration
    "initial_shape": "sinoidal", # Options: "straight", "circular" , "sinoidal"

    # Parameters for "straight" initial_shape
    "xi_pert": 0.0001, # Perturbation for straight rod (from paper, section 6.1)

    # Parameters for "circular" initial_shape
    "r0_circ_val": None, # Specify a radius value directly, e.g., 2.5.
                         # If None, calculated to make a full circle from L_eff.
    "eta_pert": 0.00,  # Perturbation for circular rod (n in Eq. 47-49 of paper, section 6.2)

    # Animation settings
    "animation_interval": 50,  # ms between frames
    "animation_steps_skip": 100, # Number of simulation steps to skip per animation frame
}
PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]

# --- Helper Functions for Regularized Stokes Flow ---

def psi_epsilon(r, epsilon_reg):
    # Eq. (12)
    if np.isscalar(r) and r < 1e-9: # Effectively zero
        return 15.0 / (8.0 * np.pi * epsilon_reg**3)
    return (15.0 * epsilon_reg**4) / (8.0 * np.pi * (r**2 + epsilon_reg**2)**(7.0/2.0))

def H1_func(r, epsilon_reg):
    # Eq. (63) / (A.19)
    return (2*epsilon_reg**2 + r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

def H2_func(r, epsilon_reg):
    # Eq. (64) / (A.20)
    return 1.0 / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

def Q_func(r, epsilon_reg):
    # Eq. (65) / (A.21)
    return (5*epsilon_reg**2 + 2*r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(5.0/2.0))

def D1_func(r, epsilon_reg):
    # Eq. (67) / (A.23)
    return (10*epsilon_reg**4 - 7*epsilon_reg**2*r**2 - 2*r**4) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))

def D2_func(r, epsilon_reg):
    # Eq. (68) / (A.24)
    return (21*epsilon_reg**2 + 6*r**2) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))


# --- Rotation Helpers ---
def get_rotation_matrix_sqrt(R_mat):
    """Computes the principal square root of a 3x3 rotation matrix."""
    try:
        # Ensure matrix is nearly a rotation matrix
        if not np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5):
            # print(f"Warning: Matrix for sqrt not perfectly SO(3). Det: {np.linalg.det(R_mat)}. R@R.T:\n{R_mat @ R_mat.T}")
            U, _, Vh = np.linalg.svd(R_mat)
            R_mat_ortho = U @ Vh
            if np.linalg.det(R_mat_ortho) < 0:
                Vh[-1,:] *= -1
                R_mat_ortho = U @ Vh
            R_mat = R_mat_ortho # Use the closest SO(3) matrix

        rot = Rotation.from_matrix(R_mat)
        rotvec = rot.as_rotvec()
        sqrt_rotvec = rotvec * 0.5
        sqrt_R_mat = Rotation.from_rotvec(sqrt_rotvec).as_matrix()
        return sqrt_R_mat
    except Exception as e:
        # Fallback for near-identity or problematic matrices
        if np.allclose(R_mat, np.eye(3)):
            return np.eye(3)
        
        # Attempt re-orthogonalization as a more robust fallback
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
            # print(f"Warning: sqrt_Ak fallback used for k. Original Ak:\n{Ak_original}")
            return sqrt_R_mat
        except Exception as final_e:
            print(f"Critical Error in get_rotation_matrix_sqrt even after fallback. Matrix:\n{R_mat}\nError: {final_e}")
            raise final_e


def get_rodrigues_rotation_matrix(axis_vector, angle_rad):
    """Computes rotation matrix using Rodrigues' formula."""
    if np.isclose(angle_rad, 0):
        return np.eye(3)
    norm_axis = np.linalg.norm(axis_vector)
    if norm_axis < 1e-9 : # No rotation if axis is zero or angle is zero
        return np.eye(3)
        
    axis_vector = axis_vector / norm_axis
    return Rotation.from_rotvec(axis_vector * angle_rad).as_matrix()

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

        # Material properties
        self.a = np.array([self.p["a1"], self.p["a2"], self.p["a3"]])
        self.b = np.array([self.p["b1"], self.p["b2"], self.p["b3"]])
        self.Omega = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])

        # Initialize rod configuration arrays
        self.X = np.zeros((self.M, 3))
        self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3))
        self.D3 = np.zeros((self.M, 3))
        
        s_vals = np.arange(self.M) * self.ds

        initial_shape = self.p.get("initial_shape", "straight")

        if initial_shape == "circular":
            # --- Circular Rod Initialization (based on Sec 6.2, Eq 46-49) ---
            if self.p.get("r0_circ_val") is not None:
                r0 = self.p["r0_circ_val"]
            elif self.L_eff > 0:
                r0 = self.L_eff / (2 * np.pi) # Default to a full circle
            else:
                r0 = 1.0 # Fallback if L_eff is zero (e.g., M=1)
            
            eta = self.p.get("eta_pert", 0.0) # Perturbation parameter (n in paper Eq 47-49)
            
            if r0 < 1e-9: # Avoid division by zero if radius is too small
                print("Warning: Circular radius r0 is very small. Defaulting to straight rod.")
                initial_shape = "straight" # Fallback to straight
            else:
                theta_s_vals = s_vals / r0 # Angle for each point s_k

                # Position X(s) = (r0 * cos(s/r0), r0 * sin(s/r0), 0) (Eq. 46)
                self.X[:, 0] = r0 * np.cos(theta_s_vals)
                self.X[:, 1] = r0 * np.sin(theta_s_vals)
                self.X[:, 2] = 0.0

                # Director D3 (tangent vector h(s/r0)) (Eq. 49)
                # D3(s) = (-sin(s/r0), cos(s/r0), 0)
                self.D3[:, 0] = -np.sin(theta_s_vals)
                self.D3[:, 1] = np.cos(theta_s_vals)
                self.D3[:, 2] = 0.0

                # Helper unit vectors for D1, D2 construction
                z_unit_vec = np.array([0.0, 0.0, 1.0]) # Global z-axis vector
                
                # Spatially varying perturbation angle: alpha_s = eta * sin(s/r0)
                alpha_s_vals = eta * np.sin(theta_s_vals) 
                cos_alpha_s = np.cos(alpha_s_vals)
                sin_alpha_s = np.sin(alpha_s_vals)

                for i in range(self.M):
                    # Radial vector r(s_i/r0) in the xy-plane
                    r_s_i_vec = np.array([np.cos(theta_s_vals[i]), np.sin(theta_s_vals[i]), 0.0])
                    
                    # Director D1 (Eq. 47)
                    # D1(s) = cos(alpha_s) * z_unit_vec + sin(alpha_s) * r_s_i_vec
                    self.D1[i, :] = cos_alpha_s[i] * z_unit_vec + sin_alpha_s[i] * r_s_i_vec
                    
                    # Director D2 (Eq. 48)
                    # D2(s) = -sin(alpha_s) * z_unit_vec + cos(alpha_s) * r_s_i_vec
                    self.D2[i, :] = -sin_alpha_s[i] * z_unit_vec + cos_alpha_s[i] * r_s_i_vec

        if initial_shape == "sinoidal":
            # Sinuoidal Rod Initialization 
            self.X[:, 0] = s_vals
            self.X[:, 1] = np.sin(s_vals / 10.0)  # Example sine wave perturbation
            self.X[:, 2] = np.cos(s_vals / 10.0)  # Example cosine wave perturbation

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

        # Ensure initial directors are orthonormal (important for stability)
        for i in range(self.M):
            d1 = self.D1[i].copy()
            d2 = self.D2[i].copy()
            
            norm_d1 = np.linalg.norm(d1)
            if norm_d1 < 1e-9: self.D1[i] = np.array([1.0,0.0,0.0]); print(f"Warning: D1[{i}] norm zero")
            else: self.D1[i] = d1 / norm_d1

            d2_ortho = d2 - np.dot(d2, self.D1[i]) * self.D1[i]
            norm_d2_ortho = np.linalg.norm(d2_ortho)
            if norm_d2_ortho < 1e-9: # D1 and D2 were parallel
                 # Create an orthogonal D2
                 temp_vec = np.array([0.0,1.0,0.0]) if np.abs(self.D1[i,0]) > 0.9 else np.array([1.0,0.0,0.0])
                 self.D2[i] = temp_vec - np.dot(temp_vec, self.D1[i]) * self.D1[i]
                 self.D2[i] /= np.linalg.norm(self.D2[i])
                 print(f"Warning: D2[{i}] re-orthogonalized due to parallelism with D1")
            else:
                 self.D2[i] = d2_ortho / norm_d2_ortho
            
            self.D3[i] = np.cross(self.D1[i], self.D2[i])
            # D3 should be unit norm by construction if D1, D2 are unit and ortho.


    def _get_D_matrix(self, k):
        """Returns the 3x3 director matrix D_k (directors as rows)."""
        return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def compute_internal_forces_and_moments(self):
        """Step 1: Compute F_{k+1/2} and N_{k+1/2}."""
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
        
        self.D_half_matrices = [] 

        for k in range(self.M - 1): 
            Dk_mat = self._get_D_matrix(k)    
            Dkp1_mat = self._get_D_matrix(k+1) 
            
            Ak = Dkp1_mat @ Dk_mat.T 
            Ak_original = Ak.copy() # For debugging if sqrt fails
            
            try:
                sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            except Exception as e:
                print(f"Error computing sqrt_Ak for segment k={k}. Ak = \n{Ak_original}")
                # Fallback: average and re-orthogonalize directors
                D_avg_d1 = (self.D1[k] + self.D1[k+1]) / 2.0
                D_avg_d2 = (self.D2[k] + self.D2[k+1]) / 2.0
                D_avg_d1 /= np.linalg.norm(D_avg_d1)
                D_avg_d2_proj_d1 = np.dot(D_avg_d2, D_avg_d1) * D_avg_d1
                D_avg_d2_ortho = D_avg_d2 - D_avg_d2_proj_d1
                if np.linalg.norm(D_avg_d2_ortho) < 1e-9: # If D1 and D2 were parallel
                    temp_vec = np.array([0.0,1.0,0.0]) if np.abs(D_avg_d1[0]) > 0.9 else np.array([1.0,0.0,0.0])
                    D_avg_d2_ortho = temp_vec - np.dot(temp_vec, D_avg_d1) * D_avg_d1
                D_avg_d2_ortho /= np.linalg.norm(D_avg_d2_ortho)

                D_avg_d3 = np.cross(D_avg_d1, D_avg_d2_ortho)
                D_half_k_mat_fallback = np.array([D_avg_d1, D_avg_d2_ortho, D_avg_d3])
                # This D_half_k_mat_fallback is D_{k+1/2} directly.
                # To use it like sqrt_Ak @ Dk_mat, we need sqrt_Ak.
                # This fallback is complex. The get_rotation_matrix_sqrt should be robust.
                # For now, re-raise if the robust get_rotation_matrix_sqrt fails.
                raise e


            D_half_k_mat = sqrt_Ak @ Dk_mat 
            self.D_half_matrices.append(D_half_k_mat)

            D1_half, D2_half, D3_half = D_half_k_mat[0], D_half_k_mat[1], D_half_k_mat[2]

            F_half_coeffs = np.zeros(3)
            dX_ds = (self.X[k+1] - self.X[k]) / self.ds
            
            F_half_coeffs[0] = self.b[0] * (np.dot(D1_half, dX_ds) - 0.0) 
            F_half_coeffs[1] = self.b[1] * (np.dot(D2_half, dX_ds) - 0.0) 
            F_half_coeffs[2] = self.b[2] * (np.dot(D3_half, dX_ds) - 1.0) 
            
            F_half[k] = F_half_coeffs[0]*D1_half + F_half_coeffs[1]*D2_half + F_half_coeffs[2]*D3_half

            N_half_coeffs = np.zeros(3)
            dD1_ds = (self.D1[k+1] - self.D1[k]) / self.ds
            dD2_ds = (self.D2[k+1] - self.D2[k]) / self.ds
            dD3_ds = (self.D3[k+1] - self.D3[k]) / self.ds

            N_half_coeffs[0] = self.a[0] * (np.dot(dD2_ds, D3_half) - self.Omega[0])
            N_half_coeffs[1] = self.a[1] * (np.dot(dD3_ds, D1_half) - self.Omega[1])
            N_half_coeffs[2] = self.a[2] * (np.dot(dD1_ds, D2_half) - self.Omega[2])

            N_half[k] = N_half_coeffs[0]*D1_half + N_half_coeffs[1]*D2_half + N_half_coeffs[2]*D3_half
            
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        """Step 2: Compute -f_k and -n_k (force/torque by fluid on rod)."""
        f_on_rod = np.zeros((self.M, 3)) 
        n_on_rod = np.zeros((self.M, 3)) 

        F_boundary_start = np.zeros(3)
        F_boundary_end = np.zeros(3)
        N_boundary_start = np.zeros(3)
        N_boundary_end = np.zeros(3)

        for k in range(self.M): 
            F_prev = F_boundary_start if k == 0 else F_half[k-1]
            F_next = F_boundary_end if k == self.M - 1 else F_half[k]
            f_on_rod[k] = (F_next - F_prev) / self.ds

            N_prev = N_boundary_start if k == 0 else N_half[k-1]
            N_next = N_boundary_end if k == self.M - 1 else N_half[k]
            dN_ds_term = (N_next - N_prev) / self.ds
            
            cross_prod_term = np.zeros(3)
            if k < self.M - 1: 
                tangent_forward = (self.X[k+1] - self.X[k]) / self.ds
                cross_prod_term += np.cross(tangent_forward, F_next) 
            
            if k > 0: 
                tangent_backward = (self.X[k] - self.X[k-1]) / self.ds
                cross_prod_term += np.cross(tangent_backward, F_prev) 
            
            n_on_rod[k] = dN_ds_term + 0.5 * cross_prod_term
            
        return f_on_rod, n_on_rod

    def compute_velocities(self, f_on_rod, n_on_rod):
        """Steps 3 & 4: Compute u(X_j) and w(X_j)."""
        u_rod = np.zeros((self.M, 3)) 
        w_rod = np.zeros((self.M, 3)) 

        g0_all = f_on_rod * self.ds
        m0_all = n_on_rod * self.ds

        for j in range(self.M): 
            Xj = self.X[j]
            sum_u_terms = np.zeros(3)
            sum_w_terms = np.zeros(3)

            for k in range(self.M): 
                Xk = self.X[k]
                g0k = g0_all[k] 
                m0k = m0_all[k] 
                
                r_vec = Xj - Xk
                r_mag = np.linalg.norm(r_vec)

                if r_mag < 1e-9: 
                    h1_val = H1_func(0.0, self.epsilon_reg)
                    # For angular velocity, D1(0) is needed for m0k contribution.
                    # Q(0) is needed for g0k contribution to angular velocity.
                    # The paper's Appendix Eq. 66 for w has (g0 x (x-X0))Q and m0 D1 + (m0.(x-X0))(x-X0)D2
                    # If r_mag is 0, (g0 x r_vec) is 0.
                    # So for w_self, only m0k term contributes from u_D.
                    d1_val_self = D1_func(0.0, self.epsilon_reg) # D1(0)
                    
                    sum_u_terms += g0k * h1_val
                    sum_w_terms += 0.25 * m0k * d1_val_self
                else:
                    h1_val = H1_func(r_mag, self.epsilon_reg)
                    h2_val = H2_func(r_mag, self.epsilon_reg)
                    q_val = Q_func(r_mag, self.epsilon_reg)
                    d1_val = D1_func(r_mag, self.epsilon_reg)
                    d2_val = D2_func(r_mag, self.epsilon_reg)

                    term_uS = g0k * h1_val + np.dot(g0k, r_vec) * r_vec * h2_val
                    term_uR_for_u = 0.5 * np.cross(m0k, r_vec) * q_val
                    sum_u_terms += term_uS + term_uR_for_u

                    term_uR_for_w = 0.5 * np.cross(g0k, r_vec) * q_val
                    term_uD = 0.25 * m0k * d1_val + 0.25 * np.dot(m0k, r_vec) * r_vec * d2_val
                    sum_w_terms += term_uR_for_w + term_uD
            
            u_rod[j] = sum_u_terms / self.mu
            w_rod[j] = sum_w_terms / self.mu
            
        return u_rod, w_rod

    def update_state(self, u_rod, w_rod):
        """Step 5: Update X_k and D_k^i."""
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


    for step in range(num_steps):
        rod.simulation_step()
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            if step % (PARAMS["animation_steps_skip"]*10) == 0 : # Print less frequently
                print(f"Step {step}/{num_steps}, Time: {step*PARAMS['dt']:.2e} s. Max X coord: {np.max(np.abs(rod.X)):.2f}")
            
            if np.max(np.abs(rod.X)) > rod.L_eff * 20 : 
                 print("Simulation unstable. Coordinates exploded.")
                 break
            if np.isnan(rod.X).any() or np.isinf(rod.X).any():
                 print("NaN/Inf detected in coordinates. Simulation unstable.")
                 break
            if np.isnan(rod.D1).any() or np.isnan(rod.D2).any() or np.isnan(rod.D3).any():
                 print("NaN detected in directors. Simulation unstable.")
                 break


    print("Simulation finished.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=3)

    if history_X:
        all_coords = np.concatenate(history_X, axis=0)
        x_min, x_max = np.min(all_coords[:,0]), np.max(all_coords[:,0])
        y_min, y_max = np.min(all_coords[:,1]), np.max(all_coords[:,1])
        z_min, z_max = np.min(all_coords[:,2]), np.max(all_coords[:,2])

        center_x, center_y, center_z = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2
        plot_range = np.max([x_max-x_min, y_max-y_min, z_max-z_min, rod.L_eff * 0.5]) * 0.6 
        if plot_range < 1e-6 : plot_range = rod.L_eff if rod.L_eff > 0 else 1.0

        ax.set_xlim([center_x - plot_range, center_x + plot_range])
        ax.set_ylim([center_y - plot_range, center_y + plot_range])
        ax.set_zlim([center_z - plot_range, center_z + plot_range])
        ax.set_aspect('equal', adjustable='box') # Attempt to make scales equal

    else: 
        ax.set_xlim([-rod.L_eff/2 if rod.L_eff > 0 else -1, rod.L_eff/2 if rod.L_eff > 0 else 1])
        ax.set_ylim([-rod.L_eff/2 if rod.L_eff > 0 else -1, rod.L_eff/2 if rod.L_eff > 0 else 1])
        ax.set_zlim([-rod.L_eff/4 if rod.L_eff > 0 else -0.5, rod.L_eff/4 if rod.L_eff > 0 else 0.5])


    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_zlabel("Z (um)")
    ax.set_title(f"Kirchhoff Rod Dynamics (Initial: {PARAMS.get('initial_shape', 'straight')})")
    ax.view_init(elev=20., azim=-35) 

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def init_animation():
        line.set_data([], [])
        line.set_3d_properties([])
        time_text.set_text('')
        return line, time_text

    def update_animation(frame_idx):
        X_data = history_X[frame_idx]
        line.set_data(X_data[:,0], X_data[:,1])
        line.set_3d_properties(X_data[:,2])
        current_time = frame_idx * PARAMS["animation_steps_skip"] * PARAMS["dt"]
        time_text.set_text(f'Time: {current_time:.3e} s')
        return line, time_text

    if history_X:
        ani = FuncAnimation(fig, update_animation, frames=len(history_X),
                            init_func=init_animation, blit=False, interval=PARAMS["animation_interval"])
        
        # --- SAVING THE ANIMATION ---
        # Option 1: Save as MP4 (requires ffmpeg)
        try:
            print("Attempting to save animation as MP4...")
            ani.save('kirchhoff_rod_animation.mp4', writer='ffmpeg', fps=15, dpi=150)
            print("Animation saved as kirchhoff_rod_animation.mp4")
        except Exception as e:
            print(f"Error saving as MP4: {e}")
            print("Make sure ffmpeg is installed and in your system's PATH.")

        # Option 2: Save as GIF (requires imagemagick or pillow)
        # Pillow writer is usually available with matplotlib, but imagemagick might give better results for complex animations.
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

        plt.tight_layout()
        plt.show() # You can still show it after saving
    else:
        print("No history data to animate. Simulation might have been too short or unstable.")

