import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba
import time

def get_fast_demo_parameters():
    """
    Reduced parameters for a FAST demonstration that runs in ~1-2 minutes.
    This allows verification of the swimming physics without the long wait.
    """
    return {
        # --- Simulation Control ---
        "dt": 1.0e-5,  # Slightly larger dt is okay with fewer points
        "total_time": 0.1, # Run for a bit longer to see movement
        
        # --- Rod Properties (Reduced for Speed) ---
        "L": 40.0,
        "M": 80,         # MUCH smaller M for O(M^2) speedup

        # --- Fluid Properties ---
        "mu": 1.0e-6,
        
        # --- Regularization ---
        "epsilon_reg_factor": 7.0,

        # --- Material Properties ---
        "a1": 3.5e-2, "a2": 3.5e-2, "a3": 3.5e-2,
        "b1": 8.0e-1, "b2": 8.0e-1, "b3": 8.0e-1,
        
        # --- Flagellar Wave Parameters ---
        "wavelength": 5.0,
        "b_amp": 0.5,
        "sigma_freq": 350,
        
        # --- Animation Settings ---
        "animation_interval": 40,
        "animation_steps_skip": 250,
    }

def get_high_fidelity_parameters():
    """
    High-fidelity parameters from the paper.
    WARNING: This simulation will take MANY HOURS to run on a standard PC.
    """
    return {
        "dt": 1.0e-6, "total_time": 0.08, "L": 40.0, "M": 600,
        "mu": 1.0e-6, "epsilon_reg_factor": 7.0, "a1": 3.5e-2, "a2": 3.5e-2,
        "a3": 3.5e-2, "b1": 8.0e-1, "b2": 8.0e-1, "b3": 8.0e-1,
        "wavelength": 5.0, "b_amp": 0.4, "sigma_freq": 350,
        "animation_interval": 40, "animation_steps_skip": 2000,
    }

# --- Helper Functions for Regularized Stokes Flow ---
@numba.njit(cache=True)
def H1_func(r, epsilon_reg): return (2*epsilon_reg**2 + r**2)/(8*np.pi*(epsilon_reg**2+r**2)**1.5)
@numba.njit(cache=True)
def H2_func(r, epsilon_reg): return 1.0/(8*np.pi*(epsilon_reg**2 + r**2)**1.5)
@numba.njit(cache=True)
def Q_func(r, epsilon_reg): return (5*epsilon_reg**2 + 2*r**2)/(8*np.pi*(epsilon_reg**2+r**2)**2.5)
@numba.njit(cache=True)
def D1_func(r, epsilon_reg): return (10*epsilon_reg**4 - 7*epsilon_reg**2*r**2 - 2*r**4)/(8*np.pi*(epsilon_reg**2+r**2)**3.5)
@numba.njit(cache=True)
def D2_func(r, epsilon_reg): return (21*epsilon_reg**2 + 6*r**2)/(8*np.pi*(epsilon_reg**2+r**2)**3.5)

# --- Rotation Helpers ---
def get_rodrigues_rotation_matrix(axis_vector, angle_rad):
    norm_axis = np.linalg.norm(axis_vector)
    if np.isclose(angle_rad, 0) or norm_axis < 1e-9: return np.eye(3)
    unit_axis = axis_vector / norm_axis
    return Rotation.from_rotvec(unit_axis * angle_rad).as_matrix()

# --- PARALLELIZED Numba JITed Core Velocity Computation ---
@numba.njit(cache=True, fastmath=True, parallel=True)
def _compute_velocities_core(M, X, g0_all, m0_all, epsilon_reg, mu):
    u_out = np.zeros((M, 3))
    w_out = np.zeros((M, 3))

    # This loop is parallelized across all available CPU cores
    for j in numba.prange(M):
        Xj = X[j]
        sum_u = np.zeros(3)
        sum_w = np.zeros(3)
        for k in range(M):
            Xk = X[k]
            g0k = g0_all[k]
            m0k = m0_all[k]
            r_vec = Xj - Xk
            r_mag_sq = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            
            if r_mag_sq < 1e-18:
                h1 = H1_func(0., epsilon_reg)
                d1 = D1_func(0., epsilon_reg)
                sum_u += g0k * h1
                sum_w += 0.25 * m0k * d1
            else:
                r_mag = np.sqrt(r_mag_sq)
                h1 = H1_func(r_mag, epsilon_reg); h2 = H2_func(r_mag, epsilon_reg)
                q = Q_func(r_mag, epsilon_reg); d1 = D1_func(r_mag, epsilon_reg)
                d2 = D2_func(r_mag, epsilon_reg)
                
                dot_g_r = np.dot(g0k, r_vec); cross_m_r = np.cross(m0k, r_vec)
                sum_u += g0k * h1 + dot_g_r * r_vec * h2 + 0.5 * cross_m_r * q
                
                dot_m_r = np.dot(m0k, r_vec); cross_g_r = np.cross(g0k, r_vec)
                sum_w += 0.5 * cross_g_r * q + 0.25 * m0k * d1 + 0.25 * dot_m_r * r_vec * d2
        
        u_out[j] = sum_u / mu
        w_out[j] = sum_w / mu
            
    return u_out, w_out

# --- The Main Rod Class ---
class KirchhoffRod:
    def __init__(self, p):
        self.p = p
        self.M, self.L, self.dt = p['M'], p['L'], p['dt']
        self.mu = p['mu']
        self.ds = self.L / (self.M - 1)
        self.s_vals = np.linspace(0, self.L, self.M)
        self.epsilon_reg = p['epsilon_reg_factor'] * self.ds
        self.a = np.array([p['a1'], p['a2'], p['a3']])
        self.b = np.array([p['b1'], p['b2'], p['b3']])
        self.k_wave = 2 * np.pi / p['wavelength']
        self.b_amp = p['b_amp']
        self.sigma_freq = p['sigma_freq']
        self.time = 0.0
        self.Omega = np.zeros((self.M, 3))

        self.X = np.zeros((self.M, 3)); self.X[:, 2] = self.s_vals
        self.D1 = np.tile([1., 0., 0.], (self.M, 1))
        self.D2 = np.tile([0., 1., 0.], (self.M, 1))
        self.D3 = np.tile([0., 0., 1.], (self.M, 1))

    def update_intrinsic_curvature(self):
        curvature_wave = self.k_wave**2 * self.b_amp * np.cos(self.k_wave * self.s_vals + self.sigma_freq * self.time)
        self.Omega[:, 0] = curvature_wave

    def compute_internal_forces_and_torques(self):
        F_half = np.zeros((self.M - 1, 3)); N_half = np.zeros((self.M - 1, 3))
        dX = np.diff(self.X, axis=0); dD1 = np.diff(self.D1, axis=0)
        dD2 = np.diff(self.D2, axis=0); dD3 = np.diff(self.D3, axis=0)

        for k in range(self.M - 1):
            D1h, D2h, D3h = 0.5 * (self.D1[k] + self.D1[k+1]), 0.5 * (self.D2[k] + self.D2[k+1]), 0.5 * (self.D3[k] + self.D3[k+1])
            D3h /= np.linalg.norm(D3h); D1h -= np.dot(D1h, D3h) * D3h
            D1h /= np.linalg.norm(D1h); D2h = np.cross(D3h, D1h)
            
            dX_ds = dX[k] / self.ds
            F_coeffs = self.b * (np.array([np.dot(D1h, dX_ds), np.dot(D2h, dX_ds), np.dot(D3h, dX_ds)]) - np.array([0,0,1]))
            F_half[k] = F_coeffs[0]*D1h + F_coeffs[1]*D2h + F_coeffs[2]*D3h

            omega_half = 0.5 * (self.Omega[k] + self.Omega[k+1])
            strain_rates = np.array([np.dot(dD2[k]/self.ds, D3h), np.dot(dD3[k]/self.ds, D1h), np.dot(dD1[k]/self.ds, D2h)])
            N_coeffs = self.a * (strain_rates - omega_half)
            N_half[k] = N_coeffs[0]*D1h + N_coeffs[1]*D2h + N_coeffs[2]*D3h
            
        return F_half, N_half

    def compute_fluid_forces(self, F_half, N_half):
        F_padded = np.vstack([np.zeros(3), F_half, np.zeros(3)])
        N_padded = np.vstack([np.zeros(3), N_half, np.zeros(3)])
        f = -(F_padded[1:] - F_padded[:-1]) / self.ds
        n = -(N_padded[1:] - N_padded[:-1]) / self.ds
        dX_ds_x_F_half = np.cross(np.diff(self.X, axis=0)/self.ds, F_half)
        cross_term_padded = np.vstack([np.zeros(3), dX_ds_x_F_half, np.zeros(3)])
        n -= 0.5 * (cross_term_padded[1:] + cross_term_padded[:-1])
        return f, n
    
    def compute_velocities(self, f, n):
        g0_all = -f * self.ds
        m0_all = -n * self.ds
        u, w = _compute_velocities_core(self.M, self.X, g0_all, m0_all, self.epsilon_reg, self.mu)
        return u, w

    def update_state(self, u_rod, w_rod):
        self.X += u_rod * self.dt
        for k in range(self.M):
            R_matrix = get_rodrigues_rotation_matrix(w_rod[k], np.linalg.norm(w_rod[k]) * self.dt)
            self.D1[k] = self.D1[k] @ R_matrix.T
            self.D2[k] = self.D2[k] @ R_matrix.T
            self.D3[k] = self.D3[k] @ R_matrix.T

    def simulation_step(self):
        self.update_intrinsic_curvature()
        F_half, N_half = self.compute_internal_forces_and_torques()
        f, n = self.compute_fluid_forces(F_half, N_half)
        u_rod, w_rod = self.compute_velocities(f, n)
        self.update_state(u_rod, w_rod)
        self.time += self.dt

# --- Main Simulation and Animation Block ---
if __name__ == '__main__':
    # --- CHOOSE WHICH SIMULATION TO RUN ---
    # PARAMS = get_high_fidelity_parameters() # WARNING: VERY SLOW (HOURS)
    PARAMS = get_fast_demo_parameters()       # USE THIS FOR A QUICK TEST (1-2 MINS)
    
    rod = KirchhoffRod(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    history_X = []
    
    print(f"Starting flagellum simulation with M={PARAMS['M']} for {PARAMS['total_time']}s.")
    print("This may take a minute or two...")
    start_time = time.time()
    
    # Run the main simulation loop
    for step in range(num_steps + 1):
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
        if step % (PARAMS["animation_steps_skip"] * 5) == 0 and step > 0:
            elapsed = time.time() - start_time
            print(f"  Step {step}/{num_steps}, Sim Time: {rod.time:.4f}s, Wall Time: {elapsed:.1f}s")
        
        if np.max(np.abs(rod.X)) > rod.L * 5 or np.isnan(rod.X).any():
            print("Simulation stopped due to instability.")
            break
        
        if step < num_steps:
            rod.simulation_step()

    print(f"Simulation finished. Total wall time: {time.time()-start_time:.2f}s.")
    
    # --- CORRECTED Animation and Saving Block ---
    if not history_X:
        print("No data to animate.")
    else:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], 'o-', lw=2, markersize=3, color='deepskyblue')
        
        ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)"); ax.set_zlabel("Z (µm)")
        ax.set_title("Flagellum Propulsion")
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        all_coords=np.concatenate(history_X,axis=0)
        current_center = np.mean(all_coords,axis=0)
        plot_half_size_z = rod.L * 0.6
        plot_half_size_xy = rod.L * 0.1
        ax.set_xlim(current_center[0] - plot_half_size_xy, current_center[0] + plot_half_size_xy)
        ax.set_ylim(current_center[1] - plot_half_size_xy, current_center[1] + plot_half_size_xy)
        ax.set_zlim(current_center[2] - plot_half_size_z, current_center[2] + plot_half_size_z)
            

        def update_animation(frame):
            X_data=history_X[frame]
            line.set_data(X_data[:,0],X_data[:,1]);line.set_3d_properties(X_data[:,2])
            time_text.set_text(f"Time: {frame*PARAMS['animation_steps_skip']*PARAMS['dt']:.3f} s")
            return line,time_text

        # Create the animation object. Using blit=False is more robust.
        ani = FuncAnimation(fig, update_animation, frames=len(history_X), blit=False, 
                            interval=PARAMS["animation_interval"])
        
        # --- Attempt to save the animation ---
        try:
            print("\nAttempting to save animation as kirchhoff_rod_animation.gif...")
            ani.save('kirchhoff_rod_animation.gif', writer='pillow', fps=25, dpi=120)
            print("-> GIF saved successfully to your script directory.")
        except Exception as e:
            print(f"\n-> Could not save animation as GIF: {e}")
            print("   Make sure 'pillow' is installed (`pip install pillow`).")

        # --- Always show the animation in a window ---
        print("\nDisplaying animation window... Please close it to exit.")
        plt.show()