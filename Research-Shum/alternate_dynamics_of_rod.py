import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

# --- Parameters ---
# These parameters are based on Tables 1 & 2 and Section 6.1 of the paper
# for an open elastic rod.

PARAMS = {
    # Rod properties
    "M": 76,  # Number of immersed boundary points
    "ds": 0.0785,  # Meshwidth for rod (um)
    # "L": 6.0, # Unstressed rod length (um) - effective length will be (M-1)*ds

    # Time integration
    "dt": 1.0e-6,  # Time step (s)
    "total_time": 0.005, # Total simulation time (s)

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

    # Intrinsic strain twist vector (um^-1) - Example from Fig. 2 top
    "Omega1": 1.3,
    "Omega2": 0.0,
    "Omega3": np.pi / 2.0,
    # "Omega1": 0.0, # For a straight rod initially relaxing if perturbed
    # "Omega2": 0.0,
    # "Omega3": 0.0,


    # Initial perturbation
    "xi_pert": 0.0001,

    # Animation settings
    "animation_interval": 50,  # ms between frames
    "animation_steps_skip": 100, # Number of simulation steps to skip per animation frame
}
PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]

# --- Helper Functions for Regularized Stokes Flow ---

def psi_epsilon(r, epsilon_reg):
    # Eq. (12)
    # Note: r can be zero here if j=k.
    # The formulas for H1, H2 etc. are well-defined at r=0.
    # This psi_epsilon is used in D1_func.
    if np.isscalar(r) and r < 1e-9: # Effectively zero
        # This case might not be strictly needed if D1 handles r=0 via Q.
        # However, for D1(0) = psi(0) - Q(0), psi(0) is needed.
        # psi_epsilon(0) = 15 / (8 * np.pi * epsilon_reg^3)
        return 15.0 / (8.0 * np.pi * epsilon_reg**3)

    return (15.0 * epsilon_reg**4) / (8.0 * np.pi * (r**2 + epsilon_reg**2)**(7.0/2.0))

# G_e, B_e and their derivatives are implicitly used in H1, H2, Q, D1, D2
# We use the direct formulas for H1, H2, Q, D1, D2 from Appendix (Eq. 63-68)

def H1_func(r, epsilon_reg):
    # Eq. (63) / (A.19)
    # Valid for r=0: H1(0) = 1 / (4 * pi * epsilon_reg)
    return (2*epsilon_reg**2 + r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

def H2_func(r, epsilon_reg):
    # Eq. (64) / (A.20)
    # Valid for r=0: H2(0) = 1 / (8 * pi * epsilon_reg^3)
    return 1.0 / (8*np.pi*(epsilon_reg**2 + r**2)**(3.0/2.0))

def Q_func(r, epsilon_reg):
    # Eq. (65) / (A.21)
    # Valid for r=0: Q(0) = 5 / (8 * pi * epsilon_reg^3)
    return (5*epsilon_reg**2 + 2*r**2) / (8*np.pi*(epsilon_reg**2 + r**2)**(5.0/2.0))

def D1_func(r, epsilon_reg):
    # Eq. (67) / (A.23)
    # D1(r) = psi_epsilon(r) - Q(r)
    # Valid for r=0: D1(0) = 5 / (4 * pi * epsilon_reg^3)
    # psi_val = psi_epsilon(r, epsilon_reg) # This is the direct definition
    # q_val = Q_func(r, epsilon_reg)
    # return psi_val - q_val
    # Direct formula from paper:
    return (10*epsilon_reg**4 - 7*epsilon_reg**2*r**2 - 2*r**4) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))


def D2_func(r, epsilon_reg):
    # Eq. (68) / (A.24)
    # Valid for r=0: D2(0) = 21 / (8 * pi * epsilon_reg^5)
    return (21*epsilon_reg**2 + 6*r**2) / \
           (8*np.pi*(epsilon_reg**2 + r**2)**(7.0/2.0))


# --- Rotation Helpers ---
def get_rotation_matrix_sqrt(R_mat):
    """Computes the principal square root of a 3x3 rotation matrix."""
    try:
        rot = Rotation.from_matrix(R_mat)
        rotvec = rot.as_rotvec()
        sqrt_rotvec = rotvec * 0.5
        sqrt_R_mat = Rotation.from_rotvec(sqrt_rotvec).as_matrix()
        return sqrt_R_mat
    except Exception as e:
        # print(f"Error in get_rotation_matrix_sqrt for matrix:\n{R_mat}\nError: {e}")
        # Fallback for near-identity or problematic matrices
        if np.allclose(R_mat, np.eye(3)):
            return np.eye(3)
        # A more robust method might be needed for edge cases,
        # but scipy usually handles them well.
        # This might indicate an issue upstream if R_mat is not a valid rotation.
        # For now, if it fails badly, return identity to avoid crash, but this is not ideal.
        # Check if trace is close to -1 (180 deg rotation)
        if np.isclose(np.trace(R_mat), -1):
             # Handle 180-degree rotation case if scipy fails
             # For a 180-degree rotation R, R*R = I. sqrt(R) is not uniquely defined in SO(3)
             # The paper implies a principal root.
             # A rotation by 90 degrees around the same axis.
             # Axis k: Rk = k. R = 2kk^T - I.
             # sqrt(R) = cos(pi/2)I + sin(pi/2) K_axis + (1-cos(pi/2))K_axis^2 = K_axis
             # This needs the axis k.
             # For now, let's re-raise if it's not a simple case.
             # print("Warning: Potentially problematic rotation matrix for sqrt.")
             # Fallback to an alternative or raise error
             # If scipy fails, it's often due to numerical precision issues making the matrix
             # slightly non-orthogonal or det != 1.
             # Try to re-orthogonalize R_mat first
            U, _, Vh = np.linalg.svd(R_mat)
            R_mat_ortho = U @ Vh
            if np.linalg.det(R_mat_ortho) < 0: # Ensure it's a proper rotation
                Vh[-1,:] *= -1
                R_mat_ortho = U @ Vh

            rot = Rotation.from_matrix(R_mat_ortho)
            rotvec = rot.as_rotvec()
            sqrt_rotvec = rotvec * 0.5
            sqrt_R_mat = Rotation.from_rotvec(sqrt_rotvec).as_matrix()
            return sqrt_R_mat


        # print(f"Warning: get_rotation_matrix_sqrt failed. Returning identity. Matrix:\n{R_mat}")
        # return np.eye(3) # Fallback, not ideal
        raise e


def get_rodrigues_rotation_matrix(axis_vector, angle_rad):
    """Computes rotation matrix using Rodrigues' formula."""
    if np.isclose(angle_rad, 0):
        return np.eye(3)
    if np.linalg.norm(axis_vector) < 1e-9 : # No rotation if axis is zero
        return np.eye(3)
        
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
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

        # Initialize rod configuration
        self.X = np.zeros((self.M, 3))
        self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3))
        self.D3 = np.zeros((self.M, 3))
        
        s_vals = np.arange(self.M) * self.ds
        xi = self.p["xi_pert"]

        # Initial configuration (Eq. 40-43)
        self.X[:, 0] = 0.0
        self.X[:, 1] = 0.0
        self.X[:, 2] = (1 + xi) * s_vals

        self.D1[:, 0] = 1.0
        self.D1[:, 1] = 0.0
        self.D1[:, 2] = 0.0

        self.D2[:, 0] = 0.0
        self.D2[:, 1] = np.cos(xi * s_vals / self.L_eff if self.L_eff > 0 else 0) # Paper uses 'xi' for perturbation, not 'xi*s' in D2/D3
                                                                                # The paper's Eq 42,43 for D2, D3 seems to be a constant rotation by xi.
                                                                                # Let's use the constant rotation from paper:
        self.D2[:, 1] = np.cos(xi)
        self.D2[:, 2] = -np.sin(xi)

        self.D3[:, 0] = 0.0
        self.D3[:, 1] = np.sin(xi)
        self.D3[:, 2] = np.cos(xi)

        # Ensure initial directors are orthonormal (they should be by construction here)
        for i in range(self.M):
            d1, d2, d3 = self.D1[i], self.D2[i], self.D3[i]
            # Gram-Schmidt if needed, but current init is orthonormal
            self.D1[i] = d1 / np.linalg.norm(d1)
            self.D2[i] = d2 - np.dot(d2, self.D1[i]) * self.D1[i]
            self.D2[i] /= np.linalg.norm(self.D2[i])
            self.D3[i] = np.cross(self.D1[i], self.D2[i])


    def _get_D_matrix(self, k):
        """Returns the 3x3 director matrix D_k (directors as rows)."""
        return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def compute_internal_forces_and_moments(self):
        """Step 1: Compute F_{k+1/2} and N_{k+1/2}."""
        # These are forces/moments on M-1 segments
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
        
        # Store D_half_matrices for use in F_half, N_half component calculation
        self.D_half_matrices = [] # list of 3x3 matrices

        for k in range(self.M - 1): # Iterate over segments
            # --- Calculate D_{k+1/2} ---
            Dk_mat = self._get_D_matrix(k)     # Directors of point k as rows
            Dkp1_mat = self._get_D_matrix(k+1) # Directors of point k+1 as rows
            
            # A_k rotates Dk to Dkp1: A_k Dk = Dkp1 => A_k = Dkp1 Dk^T
            # (since Dk is orthonormal, Dk^T = Dk^-1)
            Ak = Dkp1_mat @ Dk_mat.T 
            
            try:
                sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            except Exception as e:
                print(f"Error computing sqrt_Ak for segment k={k}. Ak = \n{Ak}")
                # If Ak is not a rotation matrix (e.g. directors became non-orthonormal)
                # this can fail.
                # Fallback: average and re-orthogonalize directors
                D_avg_d1 = (self.D1[k] + self.D1[k+1]) / 2.0
                D_avg_d2 = (self.D2[k] + self.D2[k+1]) / 2.0
                D_avg_d1 /= np.linalg.norm(D_avg_d1)
                D_avg_d2_proj_d1 = np.dot(D_avg_d2, D_avg_d1) * D_avg_d1
                D_avg_d2 = D_avg_d2 - D_avg_d2_proj_d1
                D_avg_d2 /= np.linalg.norm(D_avg_d2)
                D_avg_d3 = np.cross(D_avg_d1, D_avg_d2)
                D_half_k_mat = np.array([D_avg_d1, D_avg_d2, D_avg_d3])
                # print(f"Fallback D_half_k_mat for k={k}")
                # raise e # Or try to continue with fallback

            D_half_k_mat = sqrt_Ak @ Dk_mat # D_{k+1/2} matrix (directors as rows)
            self.D_half_matrices.append(D_half_k_mat)

            D1_half, D2_half, D3_half = D_half_k_mat[0], D_half_k_mat[1], D_half_k_mat[2]

            # --- Calculate F^i_{k+1/2} (Eq. 23) ---
            # F_half_coeffs are in the D_half basis
            F_half_coeffs = np.zeros(3)
            dX_ds = (self.X[k+1] - self.X[k]) / self.ds
            
            F_half_coeffs[0] = self.b[0] * (np.dot(D1_half, dX_ds) - 0.0) # delta_3i = 0 for i=1
            F_half_coeffs[1] = self.b[1] * (np.dot(D2_half, dX_ds) - 0.0) # delta_3i = 0 for i=2
            F_half_coeffs[2] = self.b[2] * (np.dot(D3_half, dX_ds) - 1.0) # delta_3i = 1 for i=3
            
            # Convert F_half to lab frame: F = sum F^i D^i_half
            F_half[k] = F_half_coeffs[0]*D1_half + F_half_coeffs[1]*D2_half + F_half_coeffs[2]*D3_half

            # --- Calculate N^i_{k+1/2} (Eq. 24-26) ---
            # N_half_coeffs are in the D_half basis
            N_half_coeffs = np.zeros(3)
            dD1_ds = (self.D1[k+1] - self.D1[k]) / self.ds
            dD2_ds = (self.D2[k+1] - self.D2[k]) / self.ds
            dD3_ds = (self.D3[k+1] - self.D3[k]) / self.ds

            # (i,j,k) is cyclic: (1,2,3), (2,3,1), (3,1,2)
            # N^1 = a1 * (dD2/ds . D3_half - Omega1)
            N_half_coeffs[0] = self.a[0] * (np.dot(dD2_ds, D3_half) - self.Omega[0])
            # N^2 = a2 * (dD3/ds . D1_half - Omega2)
            N_half_coeffs[1] = self.a[1] * (np.dot(dD3_ds, D1_half) - self.Omega[1])
            # N^3 = a3 * (dD1/ds . D2_half - Omega3)
            N_half_coeffs[2] = self.a[2] * (np.dot(dD1_ds, D2_half) - self.Omega[2])

            # Convert N_half to lab frame: N = sum N^i D^i_half
            N_half[k] = N_half_coeffs[0]*D1_half + N_half_coeffs[1]*D2_half + N_half_coeffs[2]*D3_half
            
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        """Step 2: Compute -f_k and -n_k (force/torque by fluid on rod)."""
        # These are forces/torques on M points
        f_on_rod = np.zeros((self.M, 3)) # -f_k in paper's notation
        n_on_rod = np.zeros((self.M, 3)) # -n_k in paper's notation

        # Boundary conditions for open rod (F_1/2 = F_M+1/2 = 0, N_1/2 = N_M+1/2 = 0)
        F_boundary_start = np.zeros(3)
        F_boundary_end = np.zeros(3)
        N_boundary_start = np.zeros(3)
        N_boundary_end = np.zeros(3)

        for k in range(self.M): # Iterate over points
            # --- Calculate -f_k (Eq. 30) ---
            F_prev = F_boundary_start if k == 0 else F_half[k-1]
            F_next = F_boundary_end if k == self.M - 1 else F_half[k]
            f_on_rod[k] = (F_next - F_prev) / self.ds

            # --- Calculate -n_k (Eq. 31) ---
            N_prev = N_boundary_start if k == 0 else N_half[k-1]
            N_next = N_boundary_end if k == self.M - 1 else N_half[k]
            dN_ds_term = (N_next - N_prev) / self.ds
            
            # Cross product term: 0.5 * ( (X_{k+1}-X_k)/ds x F_{k+1/2} + (X_k-X_{k-1})/ds x F_{k-1/2} )
            cross_prod_term = np.zeros(3)
            if k < self.M - 1: # Has F_next (F_{k+1/2})
                tangent_forward = (self.X[k+1] - self.X[k]) / self.ds
                cross_prod_term += np.cross(tangent_forward, F_next) # F_next is F_{k+1/2}
            else: # k = M-1, F_next is F_boundary_end (zero)
                pass # Term is zero
            
            if k > 0: # Has F_prev (F_{k-1/2})
                tangent_backward = (self.X[k] - self.X[k-1]) / self.ds
                cross_prod_term += np.cross(tangent_backward, F_prev) # F_prev is F_{k-1/2}
            else: # k = 0, F_prev is F_boundary_start (zero)
                pass # Term is zero
            
            n_on_rod[k] = dN_ds_term + 0.5 * cross_prod_term
            
        return f_on_rod, n_on_rod

    def compute_velocities(self, f_on_rod, n_on_rod):
        """Steps 3 & 4: Compute u(X_j) and w(X_j)."""
        u_rod = np.zeros((self.M, 3)) # Linear velocity at each point X_j
        w_rod = np.zeros((self.M, 3)) # Angular velocity at each point X_j

        # g0_all[k] = f_on_rod[k] * self.ds (this is -f_k * ds from paper)
        # m0_all[k] = n_on_rod[k] * self.ds (this is -n_k * ds from paper)
        g0_all = f_on_rod * self.ds
        m0_all = n_on_rod * self.ds

        for j in range(self.M): # Evaluation point X_j
            Xj = self.X[j]
            sum_u_terms = np.zeros(3)
            sum_w_terms = np.zeros(3)

            for k in range(self.M): # Source point X_k
                Xk = self.X[k]
                g0k = g0_all[k] # Force from source k
                m0k = m0_all[k] # Torque from source k
                
                r_vec = Xj - Xk
                r_mag = np.linalg.norm(r_vec)

                if r_mag < 1e-9: # Self-term (j == k), r_mag is effectively zero
                    h1_val = H1_func(0.0, self.epsilon_reg)
                    d1_val = D1_func(0.0, self.epsilon_reg)
                    
                    # Linear velocity self-term (from u_S)
                    # mu * u_self = g0k * H1(0)
                    sum_u_terms += g0k * h1_val
                    
                    # Angular velocity self-term (from u_D)
                    # mu * w_self = (1/4) * m0k * D1(0)
                    sum_w_terms += 0.25 * m0k * d1_val
                else:
                    h1_val = H1_func(r_mag, self.epsilon_reg)
                    h2_val = H2_func(r_mag, self.epsilon_reg)
                    q_val = Q_func(r_mag, self.epsilon_reg)
                    d1_val = D1_func(r_mag, self.epsilon_reg)
                    d2_val = D2_func(r_mag, self.epsilon_reg)

                    # Linear velocity contributions (Eq. 32, using Appendix Eq. 61 form)
                    # Term from u_S[-f_k ds] = u_S[g0k]
                    # (g0k H1) + (g0k . r_vec) r_vec H2
                    term_uS = g0k * h1_val + np.dot(g0k, r_vec) * r_vec * h2_val
                    
                    # Term from u_R[-n_k ds] = u_R[m0k]
                    # (1/2) (m0k x r_vec) Q
                    term_uR_for_u = 0.5 * np.cross(m0k, r_vec) * q_val
                    sum_u_terms += term_uS + term_uR_for_u

                    # Angular velocity contributions (Eq. 33, using Appendix Eq. 66 form)
                    # Term from u_R[-f_k ds] = u_R[g0k] (for angular vel, this is g0 x r_vec term)
                    # (1/2) (g0k x r_vec) Q
                    term_uR_for_w = 0.5 * np.cross(g0k, r_vec) * q_val
                    
                    # Term from u_D[-n_k ds] = u_D[m0k]
                    # (1/4) m0k D1 + (1/4) (m0k . r_vec) r_vec D2
                    term_uD = 0.25 * m0k * d1_val + 0.25 * np.dot(m0k, r_vec) * r_vec * d2_val
                    sum_w_terms += term_uR_for_w + term_uD
            
            u_rod[j] = sum_u_terms / self.mu
            w_rod[j] = sum_w_terms / self.mu
            
        return u_rod, w_rod

    def update_state(self, u_rod, w_rod):
        """Step 5: Update X_k and D_k^i."""
        # Update positions (Eq. 34)
        self.X += u_rod * self.dt

        # Update directors (Eq. 35)
        for k in range(self.M):
            wk = w_rod[k]
            wk_mag = np.linalg.norm(wk)
            
            if wk_mag > 1e-9: # Avoid division by zero if angular velocity is negligible
                axis_e = wk / wk_mag
                angle_theta = wk_mag * self.dt
                
                # R_matrix rotates by angle_theta around axis_e
                R_matrix = get_rodrigues_rotation_matrix(axis_e, angle_theta)
                
                # Apply rotation to D1, D2, D3
                # D_new = R @ D_old (if D_old is a column vector)
                # Here D1[k], D2[k], D3[k] are row vectors.
                # So D_new^T = R @ D_old^T => D_new = D_old @ R^T
                self.D1[k] = self.D1[k] @ R_matrix.T
                self.D2[k] = self.D2[k] @ R_matrix.T
                self.D3[k] = self.D3[k] @ R_matrix.T

                # Optional: Re-orthogonalize to prevent drift, though Rodrigues should preserve it
                # d1, d2, d3 = self.D1[k], self.D2[k], self.D3[k]
                # self.D1[k] = d1 / np.linalg.norm(d1)
                # self.D2[k] = d2 - np.dot(d2, self.D1[k]) * self.D1[k]
                # self.D2[k] /= np.linalg.norm(self.D2[k])
                # self.D3[k] = np.cross(self.D1[k], self.D2[k])


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
    print(f"Effective rod length: {rod.L_eff:.4f} um")
    print(f"Regularization epsilon: {rod.epsilon_reg:.4f} um")

    for step in range(num_steps):
        rod.simulation_step()
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            print(f"Step {step}/{num_steps}, Time: {step*PARAMS['dt']:.2e} s. Max X coord: {np.max(np.abs(rod.X)):.2f}")
            # Check for explosion
            if np.max(np.abs(rod.X)) > rod.L_eff * 10 : # If rod expands too much
                 print("Simulation unstable. Coordinates exploded.")
                 break
            if np.isnan(rod.X).any():
                 print("NaN detected in coordinates. Simulation unstable.")
                 break


    print("Simulation finished.")

    # Animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=3)

    # Determine plot limits dynamically from the first and last states, or all history
    if history_X:
        all_X_coords = np.concatenate([h[:,0] for h in history_X])
        all_Y_coords = np.concatenate([h[:,1] for h in history_X])
        all_Z_coords = np.concatenate([h[:,2] for h in history_X])

        x_min, x_max = np.min(all_X_coords), np.max(all_X_coords)
        y_min, y_max = np.min(all_Y_coords), np.max(all_Y_coords)
        z_min, z_max = np.min(all_Z_coords), np.max(all_Z_coords)

        center_x, center_y, center_z = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2
        max_range = np.max([x_max-x_min, y_max-y_min, z_max-z_min]) * 0.6 # Add some padding
        if max_range < 1e-6 : max_range = rod.L_eff # Handle static case

        ax.set_xlim([center_x - max_range, center_x + max_range])
        ax.set_ylim([center_y - max_range, center_y + max_range])
        ax.set_zlim([center_z - max_range, center_z + max_range])
    else: # Default limits if no history
        ax.set_xlim([-rod.L_eff/2, rod.L_eff/2])
        ax.set_ylim([-rod.L_eff/2, rod.L_eff/2])
        ax.set_zlim([0, rod.L_eff * 1.2])


    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_zlabel("Z (um)")
    ax.set_title("Kirchhoff Rod Dynamics")
    ax.view_init(elev=20., azim=-35) # Adjust viewing angle

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
                            init_func=init_animation, blit=False, interval=PARAMS["animation_interval"]) # blit=False for 3D usually better
        plt.tight_layout()
        plt.show()
    else:
        print("No history data to animate. Simulation might have been too short or unstable.")

