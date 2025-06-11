import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba # For performance optimization

# --- Parameters ---
PARAMS = {
    # --- SCENARIO SETUP ---
    "scenario": "gravity_fall", # Options: "twist_helix", "gravity_fall"
    
    # --- ROD PROPERTIES ---
    "M": 40,  # Number of immersed boundary points (fewer for faster gravity demo)
    "ds": 0.2, # Meshwidth for rod (um) (larger ds for a longer rod)

    # --- TIME INTEGRATION ---
    "dt": 5.0e-6,  # Time step (s) (larger dt for faster run)
    "total_time": 0.1, # Total simulation time (s) (shorter for faster run)

    # --- FLUID PROPERTIES ---
    "mu": 1.0e-6,  # Fluid viscosity (g um^-1 s^-1)
    "fluid_density_vol": 1.0e-12, # Volumetric density of fluid (g/um^3) - approx water

    # --- REGULARIZATION ---
    "epsilon_reg_factor": 4.0, # Factor for regularization parameter

    # --- MATERIAL & INTRINSIC SHAPE (dependent on scenario) ---
    "a1": 3.5e-3,  # Bending modulus
    "a2": 3.5e-3,  # Bending modulus
    "a3": 3.5e-3,  # Twist modulus
    "b1": 8.0e-1,  # Shear modulus
    "b2": 8.0e-1,  # Shear modulus
    "b3": 8.0e-1,  # Stretch modulus
    "Omega1": 0.0, # Intrinsic curvature/twist
    "Omega2": 0.0,
    "Omega3": 0.0,

    # --- INITIAL SHAPE ---
    "initial_shape": "straight", # Options: "straight", "circular"
    "straight_rod_orientation_axis": 'x', # 'x', 'y', or 'z' for initial alignment
    "xi_pert": 1e-5, # Small perturbation to break perfect symmetry

    # --- GRAVITY PARAMETERS ---
    "gravity_on": True,
    "rod_density_vol": 2.0e-12, # Rod is twice as dense as the fluid
    "rod_radius": 0.1,         # Rod cross-section radius (um)
    "gravity_vector": np.array([0.0, 0.0, -9.8e6]), # g in um/s^2

    # --- ANIMATION SETTINGS ---
    "animation_interval": 40,  # ms between frames
    "animation_steps_skip": 400, # Number of simulation steps to skip per frame (adjusted for new timing)
}

# --- Scenario-specific overrides ---
if PARAMS["scenario"] == "twist_helix":
    PARAMS.update({
        "M": 76,
        "ds": 0.0785,
        "total_time": 0.005,
        "Omega1": 1.3,
        "Omega3": np.pi / 2.0,
        "gravity_on": False,
        "straight_rod_orientation_axis": 'z',
        "animation_steps_skip": 100,
    })

PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]


# --- Helper Functions for Regularized Stokes Flow (Numba JITed) ---
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
    try:
        if not np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5):
            U, _, Vh = np.linalg.svd(R_mat)
            R_mat = U @ Vh
            if np.linalg.det(R_mat) < 0: Vh[-1,:] *= -1; R_mat = U @ Vh
        rot = Rotation.from_matrix(R_mat)
        return Rotation.from_rotvec(rot.as_rotvec() * 0.5).as_matrix()
    except Exception as e:
        print(f"Error in get_rotation_matrix_sqrt. Fallback required. Matrix:\n{R_mat}\nError: {e}")
        raise e

def get_rodrigues_rotation_matrix(axis_vector, angle_rad):
    norm_axis = np.linalg.norm(axis_vector)
    if np.isclose(angle_rad, 0) or norm_axis < 1e-9: return np.eye(3)
    return Rotation.from_rotvec(axis_vector / norm_axis * angle_rad).as_matrix()


# --- Numba JITed core velocity computation ---
@numba.njit(cache=True)
def _compute_velocities_core(M, X, g0_all, m0_all, epsilon_reg, mu):
    u_rod_out = np.zeros((M, 3)); w_rod_out = np.zeros((M, 3))
    for j in range(M):
        Xj = X[j]; sum_u = np.zeros(3); sum_w = np.zeros(3)
        for k in range(M):
            Xk = X[k]; g0k = g0_all[k]; m0k = m0_all[k]
            r_vec = Xj - Xk; r_mag_sq = r_vec[0]**2+r_vec[1]**2+r_vec[2]**2; r_mag = np.sqrt(r_mag_sq)
            if r_mag < 1e-9:
                h1 = H1_func(0.0, epsilon_reg); d1 = D1_func(0.0, epsilon_reg)
                sum_u += g0k * h1; sum_w += 0.25 * m0k * d1
            else:
                h1=H1_func(r_mag,epsilon_reg);h2=H2_func(r_mag,epsilon_reg);q=Q_func(r_mag,epsilon_reg)
                d1=D1_func(r_mag,epsilon_reg);d2=D2_func(r_mag,epsilon_reg)
                dot_g_r=np.dot(g0k,r_vec); cross_m_r=np.cross(m0k,r_vec)
                sum_u += g0k*h1 + dot_g_r*r_vec*h2 + 0.5*cross_m_r*q
                dot_m_r=np.dot(m0k,r_vec); cross_g_r=np.cross(g0k,r_vec)
                sum_w += 0.5*cross_g_r*q + 0.25*m0k*d1 + 0.25*dot_m_r*r_vec*d2
        u_rod_out[j] = sum_u / mu; w_rod_out[j] = sum_w / mu
    return u_rod_out, w_rod_out


# --- Rod Class ---
class KirchhoffRod:
    def __init__(self, params):
        self.p = params; self.M = self.p["M"]; self.ds = self.p["ds"]
        self.mu = self.p["mu"]; self.epsilon_reg = self.p["epsilon_reg"]; self.dt = self.p["dt"]
        self.a = np.array([self.p["a1"], self.p["a2"], self.p["a3"]])
        self.b = np.array([self.p["b1"], self.p["b2"], self.p["b3"]])
        self.Omega = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])
        self.X = np.zeros((self.M, 3)); self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3)); self.D3 = np.zeros((self.M, 3))
        
        # --- GRAVITY SETUP ---
        self.g_eff_force_density = np.zeros(3)
        if self.p.get("gravity_on", False):
            rod_area = np.pi * self.p["rod_radius"]**2
            delta_rho = self.p["rod_density_vol"] - self.p["fluid_density_vol"]
            self.g_eff_force_density = delta_rho * rod_area * self.p["gravity_vector"]

        # --- INITIALIZATION ---
        s_vals = np.arange(self.M) * self.ds
        if self.p["initial_shape"] == "straight":
            xi = self.p["xi_pert"]
            orient_axis = self.p.get("straight_rod_orientation_axis", 'z')
            if orient_axis == 'x':
                self.X[:, 0] = s_vals; self.D3[:, 0] = 1.0 # Tangent along X
                self.D1[:, 1] = 1.0; self.D2[:, 2] = 1.0   # D1 along Y, D2 along Z
            elif orient_axis == 'y':
                self.X[:, 1] = s_vals; self.D3[:, 1] = 1.0 # Tangent along Y
                self.D1[:, 2] = 1.0; self.D2[:, 0] = 1.0   # D1 along Z, D2 along X
            else: # Default 'z'
                self.X[:, 2] = s_vals; self.D3[:, 2] = 1.0 # Tangent along Z
                self.D1[:, 0] = 1.0; self.D2[:, 1] = 1.0   # D1 along X, D2 along Y
            # Apply small perturbation rotation to non-tangent directors
            pert_rot = get_rodrigues_rotation_matrix(self.D3[0], xi)
            self.D1 = self.D1 @ pert_rot.T
            self.D2 = self.D2 @ pert_rot.T

        # (Circular initialization code remains unchanged)
        # Final orthonormalization of initial directors
        for i in range(self.M):
            d1,d2 = self.D1[i].copy(),self.D2[i].copy(); 
            if np.linalg.norm(d1) > 1e-9: d1 /= np.linalg.norm(d1)
            else: d1 = np.array([1.,0.,0.]) # Failsafe
            d2_ortho = d2 - np.dot(d2, d1)*d1
            if np.linalg.norm(d2_ortho) > 1e-9: d2_ortho /= np.linalg.norm(d2_ortho)
            else: # Failsafe
                temp_vec = np.array([0.,1.,0.]) if np.abs(d1[0]) < 0.9 else np.array([0.,0.,1.])
                d2_ortho = np.cross(d1, temp_vec); d2_ortho /= np.linalg.norm(d2_ortho)
            self.D1[i], self.D2[i] = d1, d2_ortho; self.D3[i] = np.cross(d1, d2_ortho)

    def compute_internal_forces_and_moments(self):
        F_half = np.zeros((self.M-1,3)); N_half = np.zeros((self.M-1,3))
        for k in range(self.M-1):
            Dk_mat, Dkp1_mat = np.array([self.D1[k],self.D2[k],self.D3[k]]), np.array([self.D1[k+1],self.D2[k+1],self.D3[k+1]])
            sqrt_Ak = get_rotation_matrix_sqrt(Dkp1_mat @ Dk_mat.T)
            D_half_k_mat = sqrt_Ak @ Dk_mat
            D1h,D2h,D3h = D_half_k_mat[0],D_half_k_mat[1],D_half_k_mat[2]
            dXds = (self.X[k+1]-self.X[k])/self.ds
            F_coeffs = self.b * (np.array([np.dot(D1h,dXds),np.dot(D2h,dXds),np.dot(D3h,dXds)]) - np.array([0,0,1]))
            F_half[k] = F_coeffs[0]*D1h + F_coeffs[1]*D2h + F_coeffs[2]*D3h
            dD1ds,dD2ds,dD3ds = (self.D1[k+1]-self.D1[k])/self.ds,(self.D2[k+1]-self.D2[k])/self.ds,(self.D3[k+1]-self.D3[k])/self.ds
            N_coeffs = self.a * (np.array([np.dot(dD2ds,D3h),np.dot(dD3ds,D1h),np.dot(dD1ds,D2h)]) - self.Omega)
            N_half[k] = N_coeffs[0]*D1h + N_coeffs[1]*D2h + N_coeffs[2]*D3h
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        """Computes the force/torque exerted *by the rod on the fluid*."""
        f_on_fluid = np.zeros((self.M,3)); n_on_fluid = np.zeros((self.M,3))
        F_b_start, F_b_end, N_b_start, N_b_end = np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)
        for k in range(self.M):
            F_prev = F_b_start if k==0 else F_half[k-1]
            F_next = F_b_end if k==self.M-1 else F_half[k]
            dF_ds = (F_next - F_prev) / self.ds
            f_on_fluid[k] = dF_ds + self.g_eff_force_density # ADD GRAVITY TERM
            N_prev = N_b_start if k==0 else N_half[k-1]
            N_next = N_b_end if k==self.M-1 else N_half[k]
            dN_ds = (N_next - N_prev) / self.ds
            cross_term = np.zeros(3)
            if k<self.M-1: cross_term += np.cross((self.X[k+1]-self.X[k])/self.ds, F_next)
            if k>0: cross_term += np.cross((self.X[k]-self.X[k-1])/self.ds, F_prev)
            n_on_fluid[k] = dN_ds + 0.5 * cross_term + + self.g_eff_force_density
        return f_on_fluid, n_on_fluid

    def compute_velocities(self, f_on_fluid, n_on_fluid):
        g0_all = f_on_fluid * self.ds; m0_all = n_on_fluid * self.ds
        return _compute_velocities_core(self.M, self.X, g0_all, m0_all, self.epsilon_reg, self.mu)

    def update_state(self, u_rod, w_rod):
        self.X += u_rod * self.dt
        for k in range(self.M):
            wk = w_rod[k]
            if np.linalg.norm(wk) > 1e-9:
                R_matrix = get_rodrigues_rotation_matrix(wk, np.linalg.norm(wk)*self.dt)
                self.D1[k] = self.D1[k] @ R_matrix.T; self.D2[k] = self.D2[k] @ R_matrix.T; self.D3[k] = self.D3[k] @ R_matrix.T

    def simulation_step(self):
        F_half, N_half = self.compute_internal_forces_and_moments()
        f_on_fluid, n_on_fluid = self.compute_fluid_forces_on_rod(F_half, N_half)
        u_rod, w_rod = self.compute_velocities(f_on_fluid, n_on_fluid)
        self.update_state(u_rod, w_rod)

# --- Main Simulation and Animation Block ---
if __name__ == '__main__':
    rod = KirchhoffRod(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"]); history_X = []
    print(f"Starting scenario: '{PARAMS['scenario']}' for {PARAMS['total_time']}s.")
    import time; start_time = time.time()
    for step in range(num_steps):
        rod.simulation_step()
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            if step % (PARAMS["animation_steps_skip"]*10) == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{num_steps}, Sim Time: {step*PARAMS['dt']:.2e}s, Wall Time: {elapsed:.2f}s")
            if np.max(np.abs(rod.X)) > PARAMS["L_eff"] * 20: print("Sim unstable."); break
            if np.isnan(rod.X).any(): print("NaN in coords."); break
    print(f"Simulation finished. Total wall time: {time.time() - start_time:.2f}s.")
    
    fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([],[],[], 'o-', lw=2, markersize=4, color='c')
    ax.set_facecolor('k') # Black background
    if history_X:
        all_coords = np.concatenate(history_X,axis=0)
        center = np.mean(all_coords, axis=0)
        max_range = np.max(np.max(all_coords, axis=0) - np.min(all_coords, axis=0))
        plot_range = max_range * 0.6 if max_range > 0 else PARAMS["L_eff"]
        ax.set_xlim(center[0]-plot_range, center[0]+plot_range)
        ax.set_ylim(center[1]-plot_range, center[1]+plot_range)
        ax.set_zlim(center[2]-plot_range, center[2]+plot_range)
    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)"); ax.set_zlabel("Z (um)")
    ax.set_title(f"Kirchhoff Rod: Scenario '{PARAMS['scenario']}'")
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white')

    def update_animation(frame):
        X_data = history_X[frame]
        line.set_data(X_data[:,0], X_data[:,1]); line.set_3d_properties(X_data[:,2])
        time_text.set_text(f"Time: {frame*PARAMS['animation_steps_skip']*PARAMS['dt']:.3f} s")
        return line, time_text
    
    if history_X:
        ani = FuncAnimation(fig,update_animation,frames=len(history_X),blit=False,interval=PARAMS["animation_interval"])
        plt.show()
