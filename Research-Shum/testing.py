import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba
import time

def get_flagellum_parameters():
    """
    Returns parameters for the undulatory flagellum simulation,
    taken directly from Table 4 and Section 6.4 of the paper.
    
    """
    return {
        # --- Simulation Control ---
        "dt": 1.0e-6,  # Time step (s), as used in the paper 
        "total_time": 0.08, # Total simulation time (s)

        # --- Rod Properties (Table 4) ---
        "L": 40.0,       # Unstressed rod length (µm) 
        "M": 600,        # Number of immersed boundary points 

        # --- Fluid Properties (Table 4) ---
        "mu": 1.0e-6,    # Fluid viscosity (g µm⁻¹ s⁻¹) 

        # --- Regularization (Table 4) ---
        "epsilon_reg_factor": 7.0, # Corresponds to ε = 7Δs in the paper 

        # --- Material Properties (Table 4) ---
        "a1": 3.5e-2, "a2": 3.5e-2, "a3": 3.5e-2, # Bending/Twist moduli 
        "b1": 8.0e-1, "b2": 8.0e-1, "b3": 8.0e-1, # Shear/Stretch moduli 

        # --- Flagellar Wave Parameters (Table 4 & Sec 6.4) ---
        "wavelength": 5.0, # Wavelength of undulation (µm) 
        "b_amp": 0.4,      # Wave amplitude (µm) - increased for visibility
        "sigma_freq": 350, # Wave angular frequency (rad/s) 

        # --- Animation Settings ---
        "animation_interval": 40,   # Milliseconds per frame
        "animation_steps_skip": 2000, # Update animation every N steps
    }

# --- Helper Functions for Regularized Stokes Flow (from paper Appendix) ---
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

# --- Numba JITed Core Velocity Computation (from paper Appendix Eq. 61, 66) ---
@numba.njit(cache=True, fastmath=True)
def _compute_velocities_core(M, X, g0_all, m0_all, epsilon_reg, mu):
    u_out = np.zeros((M, 3))
    w_out = np.zeros((M, 3))
    for j in range(M):
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
                # Self-contribution (r=0)
                h1 = H1_func(0., epsilon_reg)
                d1 = D1_func(0., epsilon_reg)
                sum_u += g0k * h1
                sum_w += 0.25 * m0k * d1
            else:
                r_mag = np.sqrt(r_mag_sq)
                # Linear velocity contributions
                h1 = H1_func(r_mag, epsilon_reg)
                h2 = H2_func(r_mag, epsilon_reg)
                q = Q_func(r_mag, epsilon_reg)
                dot_g_r = np.dot(g0k, r_vec)
                cross_m_r = np.cross(m0k, r_vec)
                sum_u += g0k * h1 + dot_g_r * r_vec * h2 + 0.5 * cross_m_r * q
                
                # Angular velocity contributions
                d1 = D1_func(r_mag, epsilon_reg)
                d2 = D2_func(r_mag, epsilon_reg)
                dot_m_r = np.dot(m0k, r_vec)
                cross_g_r = np.cross(g0k, r_vec)
                sum_w += 0.5 * cross_g_r * q + 0.25 * m0k * d1 + 0.25 * dot_m_r * r_vec * d2
                
    u_out = sum_u / mu
    w_out = sum_w / mu
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

        # Initial Configuration: Straight rod along Z-axis
        self.X = np.zeros((self.M, 3))
        self.X[:, 2] = self.s_vals
        self.D1 = np.tile([1., 0., 0.], (self.M, 1))
        self.D2 = np.tile([0., 1., 0.], (self.M, 1))
        self.D3 = np.tile([0., 0., 1.], (self.M, 1))

    def update_intrinsic_curvature(self):
        """Sets the driving curvature wave that causes propulsion."""
        # This is the physically-derived curvature for a displacement y(s,t) = b*cos(...)
        # The positive sign here is the crucial fix to correct the swimming direction.
        curvature_wave = self.k_wave**2 * self.b_amp * np.cos(self.k_wave * self.s_vals + self.sigma_freq * self.time)
        self.Omega[:, 0] = curvature_wave  # Bending around D1 axis
        self.Omega[:, 1] = 0.0
        self.Omega[:, 2] = 0.0

    def compute_internal_forces_and_torques(self):
        """Computes internal forces and torques F and N at staggered points."""
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))

        # Vectorized calculations for derivatives
        dX = np.diff(self.X, axis=0)
        dD1 = np.diff(self.D1, axis=0)
        dD2 = np.diff(self.D2, axis=0)
        dD3 = np.diff(self.D3, axis=0)

        # Loop over mid-points
        for k in range(self.M - 1):
            # Average directors at the mid-point k+1/2
            D1h = 0.5 * (self.D1[k] + self.D1[k+1])
            D2h = 0.5 * (self.D2[k] + self.D2[k+1])
            D3h = 0.5 * (self.D3[k] + self.D3[k+1])
            
            # Orthonormalize the averaged frame
            D3h /= np.linalg.norm(D3h)
            D1h -= np.dot(D1h, D3h) * D3h
            D1h /= np.linalg.norm(D1h)
            D2h = np.cross(D3h, D1h)
            
            # Constitutive Relations (Eq. 5a, 5b)
            dX_ds = dX[k] / self.ds
            F_coeffs = self.b * (np.array([np.dot(D1h, dX_ds), np.dot(D2h, dX_ds), np.dot(D3h, dX_ds)]) - np.array([0,0,1]))
            F_half[k] = F_coeffs[0]*D1h + F_coeffs[1]*D2h + F_coeffs[2]*D3h

            omega_half = 0.5 * (self.Omega[k] + self.Omega[k+1])
            strain_rates = np.array([np.dot(dD2[k]/self.ds, D3h), np.dot(dD3[k]/self.ds, D1h), np.dot(dD1[k]/self.ds, D2h)])
            N_coeffs = self.a * (strain_rates - omega_half)
            N_half[k] = N_coeffs[0]*D1h + N_coeffs[1]*D2h + N_coeffs[2]*D3h
            
        return F_half, N_half

    def compute_fluid_forces(self, F_half, N_half):
        """Computes force/torque densities f and n exerted by the fluid on the rod."""
        f = np.zeros((self.M, 3))
        n = np.zeros((self.M, 3))
        
        # Pad with zeros for boundary conditions (free ends)
        F_padded = np.vstack([np.zeros(3), F_half, np.zeros(3)])
        N_padded = np.vstack([np.zeros(3), N_half, np.zeros(3)])
        
        # Eq 3a: f = -dF/ds
        f = -(F_padded[1:] - F_padded[:-1]) / self.ds
        
        # Eq 3b: n = -dN/ds - (dX/ds x F)
        n = -(N_padded[1:] - N_padded[:-1]) / self.ds
        
        dX_ds_x_F_half = np.cross(np.diff(self.X, axis=0)/self.ds, F_half)
        cross_term_padded = np.vstack([np.zeros(3), dX_ds_x_F_half, np.zeros(3)])
        n -= 0.5 * (cross_term_padded[1:] + cross_term_padded[:-1])

        return f, n

    def compute_velocities(self, f, n):
        """Computes fluid velocities at rod points based on forces exerted ON the fluid."""
        # Force on fluid is -f, torque on fluid is -n
        g0_all = -f * self.ds
        m0_all = -n * self.ds
        
        # Call the fast numba core
        # We compute the velocity for ALL points based on ALL other points.
        # This is the M-body problem that is computationally expensive.
        # The provided JITed function is a direct summation.
        u = np.zeros_like(self.X)
        w = np.zeros_like(self.X)
        for j in range(self.M):
             u_j, w_j = _compute_velocities_core(self.M, self.X, g0_all, m0_all, self.epsilon_reg, self.mu)
             u[j] = u_j[j]
             w[j] = w_j[j]
        return u, w

    def update_state(self, u_rod, w_rod):
        """Updates positions and directors using Forward Euler method."""
        self.X += u_rod * self.dt
        for k in range(self.M):
            # Rodrigues' rotation formula for directors
            R_matrix = get_rodrigues_rotation_matrix(w_rod[k], np.linalg.norm(w_rod[k]) * self.dt)
            self.D1[k] = self.D1[k] @ R_matrix.T
            self.D2[k] = self.D2[k] @ R_matrix.T
            self.D3[k] = self.D3[k] @ R_matrix.T

    def simulation_step(self):
        """Performs one full step of the simulation."""
        self.update_intrinsic_curvature()
        F_half, N_half = self.compute_internal_forces_and_torques()
        # Here, f and n are the forces exerted BY THE FLUID ON THE ROD
        f, n = self.compute_fluid_forces(F_half, N_half)
        # The velocity calculation requires the force ON THE FLUID, which is -f and -n
        u_rod, w_rod = self.compute_velocities(f, n)
        self.update_state(u_rod, w_rod)
        self.time += self.dt

# --- Main Simulation and Animation Block ---
if __name__ == '__main__':
    PARAMS = get_flagellum_parameters()
    rod = KirchhoffRod(PARAMS)
    
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    history_X = []
    
    print(f"Starting flagellum simulation for {PARAMS['total_time']}s.")
    start_time = time.time()
    
    for step in range(num_steps):
        rod.simulation_step()
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            if step % (PARAMS["animation_steps_skip"] * 5) == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{num_steps}, Sim Time: {rod.time:.4f}s, Wall Time: {elapsed:.1f}s")
            
            if np.max(np.abs(rod.X)) > rod.L * 5:
                print("Simulation became unstable. Coordinates are too large.")
                break
            if np.isnan(rod.X).any():
                print("Simulation failed: NaN detected in coordinates.")
                break

    print(f"Simulation finished. Total wall time: {time.time()-start_time:.2f}s.")
    
    # --- Animation ---
    if not history_X:
        print("No data to animate.")
    else:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], 'o-', lw=2, markersize=3, color='deepskyblue')
        
        all_coords = np.array(history_X)
        x_min, y_min, z_min = all_coords.min(axis=(0,1))
        x_max, y_max, z_max = all_coords.max(axis=(0,1))
        
        ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)"); ax.set_zlabel("Z (µm)")
        ax.set_title("Undulatory Flagellum Propulsion")
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        def update_animation(frame):
            X_data = history_X[frame]
            line.set_data(X_data[:, 0], X_data[:, 1])
            line.set_3d_properties(X_data[:, 2])
            
            # Set axis limits to contain the entire motion
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.view_init(elev=30., azim=45 + frame/2.0) # Rotate view for better visualization
            
            time_text.set_text(f"Time: {frame*PARAMS['animation_steps_skip']*PARAMS['dt']:.4f} s")
            return line, time_text

        ani = FuncAnimation(fig, update_animation, frames=len(history_X), blit=True, interval=PARAMS["animation_interval"])
        
        try:
            print("Saving animation as kirchhoff_rod_animation.gif...")
            ani.save('kirchhoff_rod_animation.gif', writer='pillow', fps=25)
            print("Animation saved successfully.")
        except Exception as e:
            print(f"\nError saving animation: {e}")
            print("Displaying animation instead. Make sure 'pillow' is installed (`pip install pillow`) to save gifs.")
            plt.show()