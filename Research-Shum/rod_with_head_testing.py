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
    "dt": 1.0e-6,  # Time step (s) - Can be slightly larger now due to better stability
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

    # --- OUTPUT SETTINGS ---
    "steps_per_history_frame": 50, # Save state every 50 steps
    "flow_viz_resolution": 40, # Grid points for flow visualization (e.g., 40x40)
    "animation_interval_ms": 50, # Frame delay for 3D animation
}

# --- Scenario-specific overrides ---
if PARAMS["scenario"] == "static_helix":
    PARAMS.update({
        "Omega1": 1.3, "Omega3": np.pi / 2.0,
        "total_time": 0.1,
    })

PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]

# --- Helper Functions (H1, H2, Q, D1, D2) ---
# These remain unchanged
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
# These remain unchanged
def get_rotation_matrix_sqrt(R_mat):
    try:
        U, _, Vh = np.linalg.svd(R_mat)
        R_mat_ortho = U @ Vh
        if np.linalg.det(R_mat_ortho) < 0:
            Vh[-1,:] *= -1
            R_mat_ortho = U @ Vh
        rot = Rotation.from_matrix(R_mat_ortho)
        rotvec = rot.as_rotvec()
        sqrt_R_mat = Rotation.from_rotvec(rotvec * 0.5).as_matrix()
        return sqrt_R_mat
    except Exception:
        if np.allclose(R_mat, np.eye(3)): return np.eye(3)
        raise
def get_rodrigues_rotation_matrix(axis_vector, angle_rad):
    if np.isclose(angle_rad, 0): return np.eye(3)
    norm_axis = np.linalg.norm(axis_vector)
    if norm_axis < 1e-9 : return np.eye(3)
    return Rotation.from_rotvec(axis_vector / norm_axis * angle_rad).as_matrix()

# --- HIGH-PERFORMANCE CORE VELOCITY COMPUTATION ---
@numba.njit(parallel=True, cache=True)
def compute_velocities_at_points_JIT(target_points, source_points, g0_sources, m0_sources, epsilon_reg, mu):
    """
    Numba-jitted and parallelized core loop for computing velocities at arbitrary
    target points due to a set of Stokeslet and Rotlet sources.
    Uses a single, uniform regularization parameter for performance.
    """
    num_targets = target_points.shape[0]
    num_sources = source_points.shape[0]
    u_out = np.zeros((num_targets, 3))
    w_out = np.zeros((num_targets, 3))

    for j in numba.prange(num_targets): # Notice parallel=True and prange
        Xj = target_points[j]
        # Private per-thread accumulators
        sum_u = np.zeros(3)
        sum_w = np.zeros(3)

        for k in range(num_sources):
            Xk, g0k, m0k = source_points[k], g0_sources[k], m0_sources[k]
            r_vec = Xj - Xk
            r_mag_sq = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            
            is_self = r_mag_sq < 1e-12
            if is_self: continue # Skip self-interaction, handle later if needed

            r_mag = np.sqrt(r_mag_sq)
            # Linear velocity contributions
            dot_g0k_rvec = g0k[0]*r_vec[0] + g0k[1]*r_vec[1] + g0k[2]*r_vec[2]
            sum_u += g0k * H1_func(r_mag, epsilon_reg) + dot_g0k_rvec * r_vec * H2_func(r_mag, epsilon_reg)
            cross_m0k_rvec = np.cross(m0k, r_vec)
            sum_u += 0.5 * cross_m0k_rvec * Q_func(r_mag, epsilon_reg)
            
            # Angular velocity contributions
            cross_g0k_rvec = np.cross(g0k, r_vec)
            sum_w += 0.5 * cross_g0k_rvec * Q_func(r_mag, epsilon_reg)
            dot_m0k_rvec = m0k[0]*r_vec[0] + m0k[1]*r_vec[1] + m0k[2]*r_vec[2]
            sum_w += 0.25 * (m0k*D1_func(r_mag, epsilon_reg) + dot_m0k_rvec*r_vec*D2_func(r_mag, epsilon_reg))

        # Add self-interaction term (approximated for target points that are also source points)
        # This is a simplification; for full accuracy, one would need to map targets to sources.
        # For now, we assume this is handled by the r_mag check.
        
        u_out[j] = sum_u / mu
        w_out[j] = sum_w / mu
            
    return u_out, w_out

# --- Kirchhoff Rod and Bacterium Classes ---
# (These are mostly unchanged except for the call to the JIT function)
class KirchhoffRod:
    def __init__(self, params):
        self.p = params; self.M = self.p["M"]; self.ds = self.p["ds"]; self.dt = self.p["dt"]
        self.a = np.array([self.p["a1"],self.p["a2"],self.p["a3"]])
        self.b = np.array([self.p["b1"],self.p["b2"],self.p["b3"]])
        self.X = np.zeros((self.M, 3)); self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3)); self.D3 = np.zeros((self.M, 3))
        self.s_vals = np.arange(self.M) * self.ds
        omega_vec = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])
        self.Omega = np.tile(omega_vec, (self.M, 1))
        self.X[:, 2] = self.s_vals; self.D3[:, 2] = 1.0; self.D1[:, 0] = 1.0; self.D2[:, 1] = 1.0
    def _get_D_matrix(self, k): return np.array([self.D1[k],self.D2[k],self.D3[k]])
    def update_intrinsic_curvature(self, time):
        if self.p["scenario"] == "flagellar_wave":
            k,b,sigma=self.p["k_wave"],self.p["b_amp"],self.p["sigma_freq"]
            self.Omega[:,0]=-k**2*b*np.sin(k*self.s_vals - sigma*time)
            self.Omega[:,1]=0.0; self.Omega[:,2]=0.0
    def compute_internal_forces_and_moments(self):
        F_half=np.zeros((self.M-1,3)); N_half=np.zeros((self.M-1,3))
        for k in range(self.M-1):
            Dk_mat, Dkp1_mat = self._get_D_matrix(k), self._get_D_matrix(k+1)
            Ak=Dkp1_mat@Dk_mat.T; sqrt_Ak=get_rotation_matrix_sqrt(Ak)
            D_half_k_mat=sqrt_Ak@Dk_mat; D1_h,D2_h,D3_h=D_half_k_mat
            dX_ds=(self.X[k+1]-self.X[k])/self.ds
            F_coeffs=self.b*(np.array([np.dot(D1_h,dX_ds),np.dot(D2_h,dX_ds),np.dot(D3_h,dX_ds)])-np.array([0,0,1]))
            F_half[k]=F_coeffs[0]*D1_h+F_coeffs[1]*D2_h+F_coeffs[2]*D3_h
            dD1ds,dD2ds,dD3ds=(self.D1[k+1]-self.D1[k])/self.ds,(self.D2[k+1]-self.D2[k])/self.ds,(self.D3[k+1]-self.D3[k])/self.ds
            N_coeffs=self.a*(np.array([np.dot(dD2ds,D3_h),np.dot(dD3ds,D1_h),np.dot(dD1ds,D2_h)])-self.Omega[k,:])
            N_half[k]=N_coeffs[0]*D1_h+N_coeffs[1]*D2_h+N_coeffs[2]*D3_h
        return F_half, N_half
    def compute_fluid_forces_on_rod(self,F_half,N_half):
        f_on_rod=np.zeros((self.M,3)); n_on_rod=np.zeros((self.M,3))
        F_b_end,N_b_end=np.zeros(3),np.zeros(3)
        for k in range(self.M):
            F_prev=F_half[k-1] if k>0 else F_half[0] # Approx for clamped end
            F_next=F_b_end if k==self.M-1 else F_half[k]
            f_on_rod[k]=(F_next-F_prev)/self.ds
            N_prev=N_half[k-1] if k>0 else N_half[0]
            N_next=N_b_end if k==self.M-1 else N_half[k]
            dN_ds=(N_next-N_prev)/self.ds
            cross_term=np.zeros(3)
            if k<self.M-1: cross_term+=np.cross((self.X[k+1]-self.X[k])/self.ds,F_next)
            if k>0: cross_term+=np.cross((self.X[k]-self.X[k-1])/self.ds,F_prev)
            n_on_rod[k]=dN_ds+0.5*cross_term
        return f_on_rod,n_on_rod
    def update_state(self,u_rod,w_rod):
        self.X+=u_rod*self.dt
        for k in range(self.M):
            wk=w_rod[k]; wk_mag=np.linalg.norm(wk)
            if wk_mag>1e-9:
                R_matrix=get_rodrigues_rotation_matrix(wk/wk_mag,wk_mag*self.dt)
                self.D1[k]=self.D1[k]@R_matrix.T; self.D2[k]=self.D2[k]@R_matrix.T
                self.D3[k]=self.D3[k]@R_matrix.T

class Bacterium:
    def __init__(self, params):
        self.p = params; self.mu = self.p["mu"]; self.dt = self.p["dt"]
        self.X_head = np.zeros(3); self.D_head = np.eye(3)
        self.r_attach_local = self.p["tail_attachment_pos"]
        self.tail = KirchhoffRod(params)
        self.time = 0.0
        self.enforce_attachment_constraint()
    def enforce_attachment_constraint(self):
        X_attach_global=self.X_head+self.D_head@self.r_attach_local
        self.tail.X+=X_attach_global-self.tail.X[0]
        self.tail.D1[0],self.tail.D2[0],self.tail.D3[0]=self.D_head
    def get_all_sources(self, f_on_rod, n_on_rod):
        g0_tail=f_on_rod*self.tail.ds; m0_tail=n_on_rod*self.tail.ds
        g0_head=-np.sum(g0_tail,axis=0)
        torque_from_tail_forces=np.sum(np.cross(self.tail.X-self.X_head,g0_tail),axis=0)
        m0_head=-(torque_from_tail_forces+np.sum(m0_tail,axis=0))
        all_source_points=np.vstack([self.X_head,self.tail.X])
        all_g0=np.vstack([g0_head,g0_tail]); all_m0=np.vstack([m0_head,m0_tail])
        return all_source_points, all_g0, all_m0
    def update_head_state(self, U_head, W_head):
        self.X_head+=U_head*self.dt; W_mag=np.linalg.norm(W_head)
        if W_mag>1e-9: self.D_head=(get_rodrigues_rotation_matrix(W_head/W_mag,W_mag*self.dt))@self.D_head
    def simulation_step(self):
        self.enforce_attachment_constraint()
        self.tail.update_intrinsic_curvature(self.time)
        F_half,N_half = self.tail.compute_internal_forces_and_moments()
        f_on_rod,n_on_rod = self.tail.compute_fluid_forces_on_rod(F_half,N_half)
        all_sources = self.get_all_sources(f_on_rod,n_on_rod)
        source_points,g0,m0 = all_sources
        target_points=np.vstack([self.X_head,self.tail.X])
        
        # *** CALL THE FAST JIT-COMPILED FUNCTION ***
        u_all,w_all = compute_velocities_at_points_JIT(
            target_points, source_points, g0, m0, self.p["epsilon_reg"], self.mu)

        U_head,W_head=u_all[0],w_all[0]
        u_tail,w_tail=u_all[1:],w_all[1:]
        self.update_head_state(U_head,W_head)
        self.tail.update_state(u_tail,w_tail)
        self.time+=self.dt
        return all_sources

# --- Flow Field Visualization (for snapshots) ---
def plot_flow_field_snapshot(ax, state):
    ax.clear()
    res=PARAMS["flow_viz_resolution"]; L_eff=PARAMS["L_eff"]
    X_head, tail_X = state['X_head'], state['tail_X']
    plot_margin = L_eff * 0.7
    x_center,y_center = X_head[0],X_head[1]
    x_grid=np.linspace(x_center-plot_margin,x_center+plot_margin,res)
    y_grid=np.linspace(y_center-plot_margin,y_center+plot_margin,res)
    gx,gy=np.meshgrid(x_grid,y_grid)
    grid_points_3d=np.vstack([gx.ravel(),gy.ravel(),np.zeros(res*res)]).T

    source_points,g0,m0 = state['sources']
    grid_velocities, _ = compute_velocities_at_points_JIT(
        grid_points_3d, source_points, g0, m0, PARAMS["epsilon_reg"], PARAMS["mu"])
    
    u_grid,v_grid=grid_velocities[:,0].reshape(res,res), grid_velocities[:,1].reshape(res,res)
    speed=np.sqrt(u_grid**2+v_grid**2)

    ax.imshow(speed, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
              origin='lower', cmap='viridis', alpha=0.8, vmin=0, vmax=np.percentile(speed, 99))
    ax.streamplot(gx,gy,u_grid,v_grid,color='white',linewidth=0.7,density=1.5)
    
    head_circle=plt.Circle((X_head[0],X_head[1]),PARAMS['head_radius'],color='red',zorder=10)
    ax.add_patch(head_circle)
    ax.plot(tail_X[:,0],tail_X[:,1],'.-',color='black',lw=2,zorder=11)
    
    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)")
    ax.set_title(f"Flow Field at t = {state['time']:.3e} s")
    ax.set_aspect('equal',adjustable='box')

# --- Main Simulation and Output Generation ---
if __name__ == '__main__':
    swimmer = Bacterium(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    history_states = []

    print(f"Starting simulation: {PARAMS['scenario']}")
    print(f"Total time: {PARAMS['total_time']}s, dt: {PARAMS['dt']}s -> {num_steps} steps.")
    start_time = time.time()

    # --- Simulation Loop ---
    for step in range(num_steps):
        all_sources = swimmer.simulation_step()
        
        if step % PARAMS["steps_per_history_frame"] == 0:
            history_states.append({
                'time': swimmer.time, 'X_head': swimmer.X_head.copy(),
                'tail_X': swimmer.tail.X.copy(), 'sources': all_sources
            })
            elapsed = time.time()-start_time
            print(f"Step {step}/{num_steps}, Sim Time: {swimmer.time:.2e} s, Wall Time: {elapsed:.2f}s")
            
        if np.isnan(swimmer.X_head).any(): print("ERROR: NaN detected."); break
    
    print(f"\n Simulation finished. Total wall time: {time.time() - start_time:.2f} seconds.")

    if not history_states:
        print("No history recorded. Exiting.")
    else:
        # --- 1. Generate and save 2D Flow Snapshots ---
        snapshot_dir = "flow_snapshots"
        if not os.path.exists(snapshot_dir): os.makedirs(snapshot_dir)
        print(f"\n  Generating 2D flow snapshots in './{snapshot_dir}'...")
        
        fig_2d, ax_2d = plt.subplots(figsize=(8,8))
        for i, state in enumerate(history_states):
            plot_flow_field_snapshot(ax_2d, state)
            filename = os.path.join(snapshot_dir, f"snapshot_{i:04d}_t_{state['time']:.4e}.png")
            fig_2d.savefig(filename, dpi=100)
        plt.close(fig_2d)
        print("Snapshots saved.")

        # --- 2. Create and save 3D Animation ---
        print("\n Generating 3D animation...")
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        tail_line, = ax_3d.plot([], [], [], 'o-', lw=2, markersize=3, color='blue')
        head_pt, = ax_3d.plot([], [], [], 'o', markersize=10, color='red')
        time_text = ax_3d.text2D(0.05, 0.95, '', transform=ax_3d.transAxes)

        all_coords = np.concatenate([s['tail_X'] for s in history_states] + [s['X_head'].reshape(1,3) for s in history_states])
        center = np.mean(all_coords, axis=0)
        max_range = np.max(np.ptp(all_coords, axis=0)) * 0.6
        ax_3d.set_xlim(center[0]-max_range, center[0]+max_range)
        ax_3d.set_ylim(center[1]-max_range, center[1]+max_range)
        ax_3d.set_zlim(center[2]-max_range, center[2]+max_range)
        ax_3d.set_xlabel("X (um)"); ax_3d.set_ylabel("Y (um)"); ax_3d.set_zlabel("Z (um)")

        def init_3d():
            tail_line.set_data_3d([], [], [])
            head_pt.set_data_3d([], [], [])
            time_text.set_text('')
            return tail_line, head_pt, time_text

        def update_3d(frame_idx):
            state = history_states[frame_idx]
            tail_X, X_head = state['tail_X'], state['X_head']
            tail_line.set_data_3d(tail_X[:,0], tail_X[:,1], tail_X[:,2])
            head_pt.set_data_3d([X_head[0]], [X_head[1]], [X_head[2]])
            time_text.set_text(f'Time: {state["time"]:.3e} s')
            return tail_line, head_pt, time_text

        ani_3d = FuncAnimation(fig_3d, update_3d, frames=len(history_states),
                               init_func=init_3d, blit=False, interval=PARAMS['animation_interval_ms'])
        
        try:
            anim_filename = f"bacterium_3d_animation_{PARAMS['scenario']}.mp4"
            ani_3d.save(anim_filename, writer='ffmpeg', fps=20, dpi=150)
            print(f"3D Animation saved as '{anim_filename}'")
        except Exception as e:
            print(f"Could not save 3D animation. Error: {e}")

        print("\nShowing interactive 3D plot...")
        plt.show()