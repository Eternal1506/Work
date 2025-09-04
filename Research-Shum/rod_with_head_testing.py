# kirchhoff_rod_with_head.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba
import time
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# --- Parameters ---
PARAMS = {
    "scenario": "flagellar_wave",
    "M": 150,
    "ds": 0.087,
    "dt": 1.0e-6,
    "total_time": 0.05,  # shorter default so example runs fast; change as needed
    "mu": 1.0e-6,
    "epsilon_reg_factor": 7.0,
    "a1": 3.5e-3, "a2": 3.5e-3, "a3": 3.5e-3,
    "b1": 8.0e-1, "b2": 8.0e-1, "b3": 8.0e-1,
    "Omega1": 0.0, "Omega2": 0.0, "Omega3": 0.0,
    "initial_shape": "straight",
    "straight_rod_orientation_axis": 'z',
    "xi_pert": 0,
    "k_wave": 2 * np.pi / 5.0,
    "b_amp": 0.2,
    "sigma_freq": 250,
    "animation_interval": 50,      # Increased interval for more complex plot
    "animation_steps_skip": 200,   # Increased skip to reduce number of frames
    "debugging": False,
    "debug_plot_interval_steps": 500,
    "save_history": True,
    # Head params
    "has_head": True,
    "R_head": 0.65,  # microns, radius of spherical head
    "head_attach_on_minus_z": False,  # attach tail base at -e3 in head frame
    # Soft spring option instead of hard attachment:
    "use_soft_attachment": False,
    "attachment_k_spring": 1e4,  # spring stiffness (if use_soft_attachment True)
    # --- NEW: Flow field visualization parameters ---
    "visualize_flow_field": True,
    "flow_grid_resolution": 35, # Number of points along each axis for the grid
}


PARAMS["L_eff"] = (PARAMS["M"] - 1) * PARAMS["ds"]
PARAMS["epsilon_reg"] = PARAMS["epsilon_reg_factor"] * PARAMS["ds"]


# --- Regularized Stokes kernel scalar functions (numba-njit) ---
@numba.njit(cache=True)
def H1_func(r, epsilon_reg):
    return (2.0 * epsilon_reg ** 2 + r ** 2) / (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (3.0 / 2.0))

@numba.njit(cache=True)
def H2_func(r, epsilon_reg):
    return 1.0 / (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (3.0 / 2.0))

@numba.njit(cache=True)
def Q_func(r, epsilon_reg):
    return (5.0 * epsilon_reg ** 2 + 2.0 * r ** 2) / (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (5.0 / 2.0))

@numba.njit(cache=True)
def D1_func(r, epsilon_reg):
    return (10.0 * epsilon_reg ** 4 - 7.0 * epsilon_reg ** 2 * r ** 2 - 2.0 * r ** 4) / \
           (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (7.0 / 2.0))

@numba.njit(cache=True)
def D2_func(r, epsilon_reg):
    return (21.0 * epsilon_reg ** 2 + 6.0 * r ** 2) / \
           (8.0 * np.pi * (epsilon_reg ** 2 + r ** 2) ** (7.0 / 2.0))


# --- Rotation Helpers ---
def get_rotation_matrix_sqrt(R_mat):
    """Computes principal square root of a 3x3 rotation matrix using scipy Rotation."""
    try:
        if not np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R_mat), 1.0, atol=1e-5):
            U, _, Vh = np.linalg.svd(R_mat)
            R_mat_ortho = U @ Vh
            if np.linalg.det(R_mat_ortho) < 0:
                Vh[-1, :] *= -1
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
    """Computes rotation matrix using Rodrigues' formula (via scipy Rotation)."""
    if np.isclose(angle_rad, 0.0):
        return np.eye(3)
    norm_axis = np.linalg.norm(axis_vector)
    if norm_axis < 1e-12:
        return np.eye(3)
    axis_vector = axis_vector / norm_axis
    return Rotation.from_rotvec(axis_vector * angle_rad).as_matrix()


# --- Numba-jitted core velocity computation supporting per-source epsilons ---
@numba.njit(cache=True)
def _compute_velocities_core(M_rod, X_rod, g0_all_rod, m0_all_rod, epsilon_reg_vec, mu_rod):
    """
    Compute linear and angular velocities for M_rod points given point forces g0_all_rod and torques m0_all_rod.
    epsilon_reg_vec is a 1D array of length M_rod giving regularization epsilon per source.
    """
    u_rod_out = np.zeros((M_rod, 3))
    w_rod_out = np.zeros((M_rod, 3))

    for j in range(M_rod):
        Xj0 = X_rod[j, 0]
        Xj1 = X_rod[j, 1]
        Xj2 = X_rod[j, 2]
        sum_u0 = 0.0
        sum_u1 = 0.0
        sum_u2 = 0.0
        sum_w0 = 0.0
        sum_w1 = 0.0
        sum_w2 = 0.0

        for k in range(M_rod):
            Xk0 = X_rod[k, 0]
            Xk1 = X_rod[k, 1]
            Xk2 = X_rod[k, 2]
            g0k0 = g0_all_rod[k, 0]
            g0k1 = g0_all_rod[k, 1]
            g0k2 = g0_all_rod[k, 2]
            m0k0 = m0_all_rod[k, 0]
            m0k1 = m0_all_rod[k, 1]
            m0k2 = m0_all_rod[k, 2]

            r0 = Xj0 - Xk0
            r1 = Xj1 - Xk1
            r2 = Xj2 - Xk2
            r_sq = r0 * r0 + r1 * r1 + r2 * r2
            r = np.sqrt(r_sq)
            eps_k = epsilon_reg_vec[k]

            if r < 1e-12:
                # self contributions: use r=0 form
                h1_val = H1_func(0.0, eps_k)
                d1_val_self = D1_func(0.0, eps_k)
                sum_u0 += g0k0 * h1_val
                sum_u1 += g0k1 * h1_val
                sum_u2 += g0k2 * h1_val

                sum_w0 += 0.25 * m0k0 * d1_val_self
                sum_w1 += 0.25 * m0k1 * d1_val_self
                sum_w2 += 0.25 * m0k2 * d1_val_self
            else:
                h1_val = H1_func(r, eps_k)
                h2_val = H2_func(r, eps_k)
                q_val = Q_func(r, eps_k)
                d1_val = D1_func(r, eps_k)
                d2_val = D2_func(r, eps_k)

                # dot g0k . r_vec
                dot_g0k_r = g0k0 * r0 + g0k1 * r1 + g0k2 * r2
                # u_S term
                sum_u0 += g0k0 * h1_val + dot_g0k_r * r0 * h2_val
                sum_u1 += g0k1 * h1_val + dot_g0k_r * r1 * h2_val
                sum_u2 += g0k2 * h1_val + dot_g0k_r * r2 * h2_val

                # rotlet contribution to u: 0.5 (m x r) Q
                cross_mr0 = m0k1 * r2 - m0k2 * r1
                cross_mr1 = m0k2 * r0 - m0k0 * r2
                cross_mr2 = m0k0 * r1 - m0k1 * r0

                sum_u0 += 0.5 * cross_mr0 * q_val
                sum_u1 += 0.5 * cross_mr1 * q_val
                sum_u2 += 0.5 * cross_mr2 * q_val

                # angular velocity contributions
                cross_g_r0 = g0k1 * r2 - g0k2 * r1
                cross_g_r1 = g0k2 * r0 - g0k0 * r2
                cross_g_r2 = g0k0 * r1 - g0k1 * r0

                sum_w0 += 0.5 * cross_g_r0 * q_val
                sum_w1 += 0.5 * cross_g_r1 * q_val
                sum_w2 += 0.5 * cross_g_r2 * q_val

                # u_D term: 1/4 m0k D1 + 1/4 (m0k·r) r D2
                dot_m0k_r = m0k0 * r0 + m0k1 * r1 + m0k2 * r2
                sum_w0 += 0.25 * m0k0 * d1_val + 0.25 * dot_m0k_r * r0 * d2_val
                sum_w1 += 0.25 * m0k1 * d1_val + 0.25 * dot_m0k_r * r1 * d2_val
                sum_w2 += 0.25 * m0k2 * d1_val + 0.25 * dot_m0k_r * r2 * d2_val

        u_rod_out[j, 0] = sum_u0 / mu_rod
        u_rod_out[j, 1] = sum_u1 / mu_rod
        u_rod_out[j, 2] = sum_u2 / mu_rod

        w_rod_out[j, 0] = sum_w0 / mu_rod
        w_rod_out[j, 1] = sum_w1 / mu_rod
        w_rod_out[j, 2] = sum_w2 / mu_rod

    return u_rod_out, w_rod_out


# --- Numba-jitted evaluator for velocity field on arbitrary evaluation points ---
@numba.njit(cache=True)
def _evaluate_velocity_field_core(eval_pts, X_sources, g0_sources, m0_sources, eps_sources, mu):
    """
    eval_pts: (Ne,3)
    X_sources: (Ns,3)
    g0_sources: (Ns,3)
    m0_sources: (Ns,3)
    eps_sources: (Ns,)
    returns u_eval (Ne,3)
    """
    Ne = eval_pts.shape[0]
    Ns = X_sources.shape[0]
    u_eval = np.zeros((Ne, 3))

    for i in range(Ne):
        xi0 = eval_pts[i, 0]
        xi1 = eval_pts[i, 1]
        xi2 = eval_pts[i, 2]
        sum0 = 0.0
        sum1 = 0.0
        sum2 = 0.0
        for k in range(Ns):
            Xk0 = X_sources[k, 0]
            Xk1 = X_sources[k, 1]
            Xk2 = X_sources[k, 2]
            gk0 = g0_sources[k, 0]
            gk1 = g0_sources[k, 1]
            gk2 = g0_sources[k, 2]
            mk0 = m0_sources[k, 0]
            mk1 = m0_sources[k, 1]
            mk2 = m0_sources[k, 2]

            r0 = xi0 - Xk0
            r1 = xi1 - Xk1
            r2 = xi2 - Xk2
            r_sq = r0 * r0 + r1 * r1 + r2 * r2
            r = np.sqrt(r_sq)
            eps_k = eps_sources[k]

            if r < 1e-12:
                h1_val = H1_func(0.0, eps_k)
                sum0 += gk0 * h1_val
                sum1 += gk1 * h1_val
                sum2 += gk2 * h1_val
            else:
                h1_val = H1_func(r, eps_k)
                h2_val = H2_func(r, eps_k)
                q_val = Q_func(r, eps_k)

                dot_g_r = gk0 * r0 + gk1 * r1 + gk2 * r2
                sum0 += gk0 * h1_val + dot_g_r * r0 * h2_val
                sum1 += gk1 * h1_val + dot_g_r * r1 * h2_val
                sum2 += gk2 * h1_val + dot_g_r * r2 * h2_val

                cross_mr0 = mk1 * r2 - mk2 * r1
                cross_mr1 = mk2 * r0 - mk0 * r2
                cross_mr2 = mk0 * r1 - mk1 * r0

                sum0 += 0.5 * cross_mr0 * q_val
                sum1 += 0.5 * cross_mr1 * q_val
                sum2 += 0.5 * cross_mr2 * q_val

        u_eval[i, 0] = sum0 / mu
        u_eval[i, 1] = sum1 / mu
        u_eval[i, 2] = sum2 / mu

    return u_eval


# --- Rod Class ---
class KirchhoffRod:
    def __init__(self, params):
        self.p = params
        self.M = self.p["M"]
        self.ds = self.p["ds"]
        self.mu = self.p["mu"]
        self.epsilon_reg = self.p["epsilon_reg"]
        self.dt = self.p["dt"]
        self.L_eff = self.p["L_eff"]
        self.a = np.array([self.p["a1"], self.p["a2"], self.p["a3"]])
        self.b = np.array([self.p["b1"], self.p["b2"], self.p["b3"]])

        self.X = np.zeros((self.M, 3))
        self.D1 = np.zeros((self.M, 3))
        self.D2 = np.zeros((self.M, 3))
        self.D3 = np.zeros((self.M, 3))

        self.time = 0.0
        self.s_vals = np.arange(self.M) * self.ds

        omega_vec = np.array([self.p["Omega1"], self.p["Omega2"], self.p["Omega3"]])
        self.Omega = np.tile(omega_vec, (self.M, 1))

        # Head settings
        self.has_head = bool(self.p.get("has_head", False))
        if self.has_head:
            self.R_head = float(self.p.get("R_head", 0.65))
            self.eps_head = float(self.R_head)
            # Head center world position
            self.X_head = np.zeros(3)
            # Head orientation E_head: columns are body axes in world frame
            self.E_head = np.eye(3)
            # attachment offset in head body frame
            if self.p.get("head_attach_on_minus_z", True):
                self.r_attach_head = np.array([0.0, 0.0, -self.R_head])
            else:
                self.r_attach_head = np.array([0.0, 0.0, self.R_head])
            self.enforce_attachment = True
            self.use_soft_attachment = bool(self.p.get("use_soft_attachment", False))
            self.attachment_k_spring = float(self.p.get("attachment_k_spring", 1e4))

        # --- Initial Shape Setup ---
        initial_shape = self.p.get("initial_shape", "straight")
        xi = self.p.get("xi_pert", 0.0)
        orient_axis = self.p.get("straight_rod_orientation_axis", 'z')
        if initial_shape == "straight":
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
            else:
                self.X[:, 2] = self.s_vals
                self.D3[:, 2] = 1.0
                self.D1[:, 0] = 1.0
                self.D2[:, 1] = 1.0

            # apply small perturbation rotation about tangent if desired
            pert_rot = get_rodrigues_rotation_matrix(self.D3[0], xi)
            # Rotate director frames accordingly
            for i in range(self.M):
                self.D1[i] = (pert_rot @ self.D1[i])
                self.D2[i] = (pert_rot @ self.D2[i])
                self.D3[i] = (pert_rot @ self.D3[i])

        # Gram-Schmidt ensure orthonormal directors
        for i in range(self.M):
            d1 = self.D1[i].copy()
            n1 = np.linalg.norm(d1)
            if n1 < 1e-12:
                self.D1[i] = np.array([1.0, 0.0, 0.0])
            else:
                self.D1[i] = d1 / n1
            d2 = self.D2[i].copy()
            d2_ortho = d2 - np.dot(d2, self.D1[i]) * self.D1[i]
            n2 = np.linalg.norm(d2_ortho)
            if n2 < 1e-12:
                temp_vec = np.array([0.0, 1.0, 0.0]) if np.abs(self.D1[i, 0]) > 0.9 else np.array([1.0, 0.0, 0.0])
                self.D2[i] = temp_vec - np.dot(temp_vec, self.D1[i]) * self.D1[i]
                self.D2[i] /= np.linalg.norm(self.D2[i])
            else:
                self.D2[i] = d2_ortho / n2
            self.D3[i] = np.cross(self.D1[i], self.D2[i])

        # place head so that the first rod node attaches at the correct spot
        if self.has_head:
            attach_world = self.X[0]
            self.E_head = np.eye(3)
            self.X_head = attach_world - (self.E_head @ self.r_attach_head)

    def _get_D_matrix(self, k):
        return np.array([self.D1[k], self.D2[k], self.D3[k]]).T # Return as columns

    def update_intrinsic_curvature(self):
        if self.p["scenario"] == "flagellar_wave":
            k_wave = self.p["k_wave"]
            b_amp = self.p["b_amp"]
            sigma_freq = self.p["sigma_freq"]
            self.Omega[:, 0] = -k_wave**2 * b_amp * np.cos(k_wave * self.s_vals) * np.sin(sigma_freq * self.time)
            self.Omega[:, 1] = k_wave**2 * b_amp * np.sin(k_wave * self.s_vals) * np.sin(sigma_freq * self.time)
            self.Omega[:, 2] = 0.0


    def compute_internal_forces_and_moments(self):
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
        self.D_half_matrices = []
        for k in range(self.M - 1):
            Dk_mat = self._get_D_matrix(k)
            Dkp1_mat = self._get_D_matrix(k + 1)
            Ak = Dkp1_mat.T @ Dk_mat
            sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            D_half_k_mat = Dk_mat @ sqrt_Ak
            self.D_half_matrices.append(D_half_k_mat)
            D1_h, D2_h, D3_h = D_half_k_mat[:, 0], D_half_k_mat[:, 1], D_half_k_mat[:, 2]

            dX_ds = (self.X[k + 1] - self.X[k]) / self.ds
            F_coeffs = np.array([
                self.b[0] * (np.dot(D1_h, dX_ds)),
                self.b[1] * (np.dot(D2_h, dX_ds)),
                self.b[2] * (np.dot(D3_h, dX_ds) - 1.0)
            ])
            F_half[k] = F_coeffs[0] * D1_h + F_coeffs[1] * D2_h + F_coeffs[2] * D3_h
            
            dD1ds = (self.D1[k + 1] - self.D1[k]) / self.ds
            dD2ds = (self.D2[k + 1] - self.D2[k]) / self.ds
            dD3ds = (self.D3[k + 1] - self.D3[k]) / self.ds
            
            # Curvatures kappa_i = dD_j/ds . D_k (cyclic i,j,k)
            kappa1 = np.dot(dD2ds, D3_h)
            kappa2 = np.dot(dD3ds, D1_h)
            kappa3 = np.dot(dD1ds, D2_h)

            N_coeffs = np.array([
                self.a[0] * (kappa1 - self.Omega[k, 0]),
                self.a[1] * (kappa2 - self.Omega[k, 1]),
                self.a[2] * (kappa3 - self.Omega[k, 2])
            ])
            N_half[k] = N_coeffs[0] * D1_h + N_coeffs[1] * D2_h + N_coeffs[2] * D3_h
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        f_on_rod = np.zeros((self.M, 3))
        n_on_rod = np.zeros((self.M, 3))
        F_b_start, F_b_end = np.zeros(3), np.zeros(3)
        N_b_start, N_b_end = np.zeros(3), np.zeros(3)
        
        for k in range(self.M):
            F_prev = F_b_start if k == 0 else F_half[k - 1]
            F_next = F_b_end if k == self.M - 1 else F_half[k]
            f_on_rod[k] = (F_next - F_prev) / self.ds

            N_prev = N_b_start if k == 0 else N_half[k - 1]
            N_next = N_b_end if k == self.M - 1 else N_half[k]
            dN_ds = (N_next - N_prev) / self.ds

            cross_term = np.zeros(3)
            # Use centered difference for tangent vector X_s
            if k == 0:
                X_s = (self.X[1] - self.X[0]) / self.ds
            elif k == self.M - 1:
                X_s = (self.X[k] - self.X[k-1]) / self.ds
            else:
                X_s = (self.X[k+1] - self.X[k-1]) / (2 * self.ds)
            
            n_on_rod[k] = dN_ds + np.cross(X_s, F_half[k-1] if k > 0 else F_b_start) # Simplified, better to average F

        if self.has_head and self.use_soft_attachment:
            attach_world = self.X_head + self.E_head @ self.r_attach_head
            delta = self.X[0] - attach_world
            f_spring = -self.attachment_k_spring * delta
            f_on_rod[0] += f_spring / self.ds
        return f_on_rod, n_on_rod

    def compute_head_force_and_torque(self, f_on_rod, n_on_rod):
        total_force = np.sum(f_on_rod * self.ds, axis=0)
        rel = self.X - self.X_head
        cross_terms = np.cross(rel, f_on_rod * self.ds)
        total_torque = np.sum(n_on_rod * self.ds, axis=0) + np.sum(cross_terms, axis=0)
        g0_head = -total_force
        m0_head = -total_torque
        return g0_head, m0_head

    def compute_velocities(self, f_on_rod, n_on_rod):
        g0_fil = f_on_rod * self.ds
        m0_fil = n_on_rod * self.ds

        if not self.has_head:
            eps_vec = np.full(self.M, self.epsilon_reg)
            u_all, w_all = _compute_velocities_core(self.M, self.X, g0_fil, m0_fil, eps_vec, self.mu)
            return u_all, w_all, None, None
        else:
            g0_head, m0_head = self.compute_head_force_and_torque(f_on_rod, n_on_rod)
            X_all = np.vstack([self.X, self.X_head.reshape(1, 3)])
            g0_all = np.vstack([g0_fil, g0_head.reshape(1, 3)])
            m0_all = np.vstack([m0_fil, m0_head.reshape(1, 3)])
            eps_vec = np.append(np.full(self.M, self.epsilon_reg), self.eps_head)
            Mtot = self.M + 1
            u_all, w_all = _compute_velocities_core(Mtot, X_all, g0_all, m0_all, eps_vec, self.mu)
            return u_all[:self.M, :], w_all[:self.M, :], u_all[self.M, :], w_all[self.M, :]

    def update_state(self, u_rod, w_rod):
        self.X += u_rod * self.dt
        for k in range(self.M):
            wk = w_rod[k]
            wk_mag = np.linalg.norm(wk)
            if wk_mag > 1e-12:
                axis_e = wk / wk_mag
                angle_theta = wk_mag * self.dt
                R_matrix = get_rodrigues_rotation_matrix(axis_e, angle_theta)
                self.D1[k] = (R_matrix @ self.D1[k])
                self.D2[k] = (R_matrix @ self.D2[k])
                self.D3[k] = (R_matrix @ self.D3[k])

    def update_head_pose(self, u_head, w_head):
        if not self.has_head: return
        self.X_head += u_head * self.dt
        wmag = np.linalg.norm(w_head)
        if wmag > 1e-12:
            R = get_rodrigues_rotation_matrix(w_head / wmag, wmag * self.dt)
            self.E_head = R @ self.E_head
        if self.enforce_attachment and not self.use_soft_attachment:
            attach_world = self.X_head + self.E_head @ self.r_attach_head
            self.X[0] = attach_world.copy()
            # Note: aligning directors is a strong constraint, can be optional
            self.D3[0] = self.E_head[:, 2].copy()
            self.D1[0] = self.E_head[:, 0].copy()
            self.D2[0] = self.E_head[:, 1].copy()

    def simulation_step(self):
        self.update_intrinsic_curvature()
        F_half, N_half = self.compute_internal_forces_and_moments()
        f_on_rod, n_on_rod = self.compute_fluid_forces_on_rod(F_half, N_half)
        u_rod, w_rod, u_head, w_head = self.compute_velocities(f_on_rod, n_on_rod)
        self.update_state(u_rod, w_rod)
        if self.has_head and u_head is not None:
            self.update_head_pose(u_head, w_head)
        self.time += self.dt
        return u_rod, w_rod, f_on_rod, n_on_rod

# --- Standalone evaluate_velocity_field function using numba core ---
def evaluate_velocity_field(eval_points, X_sources, g0_sources, m0_sources, eps_sources, mu):
    eval_pts = np.ascontiguousarray(eval_points.astype(np.float64))
    Xs = np.ascontiguousarray(X_sources.astype(np.float64))
    g0s = np.ascontiguousarray(g0_sources.astype(np.float64))
    m0s = np.ascontiguousarray(m0_sources.astype(np.float64))
    epss = np.ascontiguousarray(eps_sources.astype(np.float64))
    return _evaluate_velocity_field_core(eval_pts, Xs, g0s, m0s, epss, mu)

# -------------------------------
# Main simulation + animation
# -------------------------------
if __name__ == '__main__':
    rod = KirchhoffRod(PARAMS)
    num_steps = int(PARAMS["total_time"] / PARAMS["dt"])
    history_X, history_u, history_w, history_f, history_n = [], [], [], [], []
    history_X_head = []
    
    print(f"Starting simulation for {PARAMS['total_time']}s, with dt={PARAMS['dt']}s ({num_steps} steps).")
    print(f"Rod initial shape: {PARAMS.get('initial_shape', 'straight')}")
    print(f"Effective rod length: {rod.L_eff:.4f} um")
    if rod.has_head: print(f"Head radius: {rod.R_head:.4f} um")
    
    start_time = time.time()
    
    for step in range(num_steps + 1):
        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            history_f.append(None) # Placeholder
            history_n.append(None)
            history_u.append(None)
            history_w.append(None)
            if rod.has_head:
                history_X_head.append(rod.X_head.copy())
            
            # Recompute forces/velocities for this snapshot to store them
            rod.update_intrinsic_curvature()
            F_half, N_half = rod.compute_internal_forces_and_moments()
            f_on_rod, n_on_rod = rod.compute_fluid_forces_on_rod(F_half, N_half)
            u_rod, w_rod, _, _ = rod.compute_velocities(f_on_rod, n_on_rod)
            history_f[-1] = f_on_rod.copy()
            history_n[-1] = n_on_rod.copy()
            history_u[-1] = u_rod.copy()
            history_w[-1] = w_rod.copy()

            if step % (PARAMS["animation_steps_skip"] * 10) == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{num_steps}, Sim Time: {rod.time:.2e} s, Wall Time: {elapsed:.2f}s.")
        
        if np.max(np.abs(rod.X)) > rod.L_eff * 50:
            print("Simulation unstable. Coordinates exploded.")
            break
        if np.isnan(rod.X).any() or np.isnan(rod.D1).any():
            print("NaN detected. Simulation unstable.")
            break
        
        if step < num_steps:
             rod.simulation_step() # Don't step after the last save

    end_time = time.time()
    print(f"Simulation finished. Total wall time: {end_time - start_time:.2f} seconds.")
    
    if not history_X:
        print("No history data to process. Exiting.")
        exit()

    # --- Velocity Continuity Check at the Final Step ---
    print("\n--- Performing velocity continuity check at final step ---")
    X_final_rod = history_X[-1]
    f_final = history_f[-1]
    n_final = history_n[-1]
    u_from_sim = history_u[-1]
    
    g0_final_rod = f_final * rod.ds
    m0_final_rod = n_final * rod.ds
    eps_final_rod = np.full(rod.M, rod.epsilon_reg)

    if rod.has_head:
        X_final_head = history_X_head[-1]
        # Recalculate head force/torque for final frame
        total_force = np.sum(g0_final_rod, axis=0)
        rel = X_final_rod - X_final_head
        total_torque = np.sum(m0_final_rod, axis=0) + np.sum(np.cross(rel, g0_final_rod), axis=0)
        g0_head = -total_force
        m0_head = -total_torque

        X_sources = np.vstack([X_final_rod, X_final_head.reshape(1,3)])
        g0_sources = np.vstack([g0_final_rod, g0_head.reshape(1,3)])
        m0_sources = np.vstack([m0_final_rod, m0_head.reshape(1,3)])
        eps_sources = np.append(eps_final_rod, rod.eps_head)
    else:
        X_sources, g0_sources, m0_sources, eps_sources = X_final_rod, g0_final_rod, m0_final_rod, eps_final_rod

    u_evaluated_at_nodes = evaluate_velocity_field(X_final_rod, X_sources, g0_sources, m0_sources, eps_sources, rod.mu)
    
    max_abs_diff = np.max(np.abs(u_evaluated_at_nodes - u_from_sim))
    print(f"Max absolute difference between simulated and evaluated velocities at nodes: {max_abs_diff:.2e}")
    if max_abs_diff < 1e-9:
        print(" Check PASSED: Velocities are continuous and consistent.")
    else:
        print(" Check FAILED: Velocities are inconsistent.")
    print("-------------------------------------------------------\n")
    
    # --- Animation Setup ---
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    fig.suptitle("Waving Filament Simulation and Flow Field")

    line_3d, = ax1.plot([], [], [], 'o-', lw=2, markersize=3, color='black')
    head_3d, = ax1.plot([], [], [], 'o', color='red', markersize=10)
    time_text_3d = ax1.text2D(0.05, 0.95, '', transform=ax1.transAxes)

    # Determine plot limits from all historical data
    all_coords = np.concatenate(history_X, axis=0)
    center = np.mean(all_coords, axis=0)
    max_range = np.max(np.ptp(all_coords, axis=0))
    plot_range = max(max_range, rod.L_eff) * 1.0 # Add padding

    # Setup 3D plot
    ax1.set_xlim([center[0] - plot_range/2, center[0] + plot_range/2])
    ax1.set_ylim([center[1] - plot_range/2, center[1] + plot_range/2])
    ax1.set_zlim([center[2] - plot_range/2, center[2] + plot_range/2])
    ax1.set_xlabel("X (μm)"); ax1.set_ylabel("Y (μm)"); ax1.set_zlabel("Z (μm)")
    ax1.set_title("3D Filament Dynamics")
    ax1.view_init(elev=20., azim=-35)
    try: ax1.set_aspect('equal', adjustable='box')
    except NotImplementedError: pass

    # --- Setup for Flow Field Visualization ---
    if PARAMS["visualize_flow_field"]:
        res = PARAMS["flow_grid_resolution"]
        gx = np.linspace(center[0] - plot_range / 2, center[0] + plot_range / 2, res)
        gz = np.linspace(center[2] - plot_range / 2, center[2] + plot_range / 2, res)
        gxx, gzz = np.meshgrid(gx, gz)
        eval_points_grid = np.vstack([gxx.ravel(), np.full_like(gxx.ravel(), center[1]), gzz.ravel()]).T

        # Single, persistent colorbar (no per-frame creation!)
        norm = Normalize(vmin=0.0, vmax=1.0)   # vmax will be updated per frame
        sm = ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])                       # required by colorbar
        cb = fig.colorbar(sm, ax=ax2, label='Velocity Magnitude (μm/s)')


    def update_animation(frame_idx):
        # --- Update 3D Plot ---
        X_data = history_X[frame_idx]
        line_3d.set_data(X_data[:, 0], X_data[:, 1])
        line_3d.set_3d_properties(X_data[:, 2])
        current_time = frame_idx * PARAMS["animation_steps_skip"] * PARAMS["dt"]
        time_text_3d.set_text(f'Time: {current_time:.3e} s')
        
        if rod.has_head:
            Xh = history_X_head[frame_idx]
            head_3d.set_data([Xh[0]], [Xh[1]]); head_3d.set_3d_properties([Xh[2]])

        # --- Update 2D Flow Plot ---
        ax2.cla()
        if PARAMS["visualize_flow_field"]:
            f_rod, n_rod = history_f[frame_idx], history_n[frame_idx]
            g0_rod, m0_rod = f_rod * rod.ds, n_rod * rod.ds
            
            # Assemble all sources for the current frame
            if rod.has_head:
                Xh = history_X_head[frame_idx]
                total_force = np.sum(g0_rod, axis=0)
                rel = X_data - Xh
                total_torque = np.sum(m0_rod, axis=0) + np.sum(np.cross(rel, g0_rod), axis=0)
                g0_head, m0_head = -total_force, -total_torque
                
                X_srcs = np.vstack([X_data, Xh.reshape(1,3)])
                g0_srcs = np.vstack([g0_rod, g0_head.reshape(1,3)])
                m0_srcs = np.vstack([m0_rod, m0_head.reshape(1,3)])
                eps_srcs = np.append(np.full(rod.M, rod.epsilon_reg), rod.eps_head)
            else:
                X_srcs, g0_srcs, m0_srcs, eps_srcs = X_data, g0_rod, m0_rod, np.full(rod.M, rod.epsilon_reg)

            # Evaluate velocity on the grid and reshape
            u_grid = evaluate_velocity_field(eval_points_grid, X_srcs, g0_srcs, m0_srcs, eps_srcs, rod.mu)
            u_x = u_grid[:, 0].reshape(gxx.shape)
            u_z = u_grid[:, 2].reshape(gzz.shape)
            speed = np.sqrt(u_x**2 + u_z**2)

            # Update the color scaling (use 99th percentile for stability)
            new_vmax = float(np.nanpercentile(speed, 99))
            if new_vmax <= 0:
                new_vmax = 1.0
            norm.vmax = new_vmax
            cb.update_normal(sm)  # refresh the colorbar scale

            # Draw streamlines using the shared norm/cmap (NO new colorbar here)
            strm = ax2.streamplot(gxx, gzz, u_x, u_z, color=speed, cmap='viridis',
                                norm=norm, density=1.5, linewidth=1)

        
        # Plot rod projection on 2D plot
        ax2.plot(X_data[:, 0], X_data[:, 2], 'o-', color='black', lw=3, markersize=4, zorder=10)
        if rod.has_head:
             ax2.plot(Xh[0], Xh[2], 'o', color='red', markersize=12, zorder=11)

        ax2.set_title(f"Flow Field (X-Z Plane)")
        ax2.set_xlabel("X (μm)"); ax2.set_ylabel("Z (μm)")
        ax2.set_xlim(gx[0], gx[-1]); ax2.set_ylim(gz[0], gz[-1])
        ax2.set_aspect('equal', adjustable='box')
        
        return line_3d, head_3d, time_text_3d

    ani = FuncAnimation(fig, update_animation, frames=len(history_X),
                          blit=False, interval=PARAMS["animation_interval"])
    
    try:
        save_filename = 'kirchhoff_rod_flow_animation.mp4'
        print(f"Attempting to save animation as {save_filename}...")
        ani.save(save_filename, writer='ffmpeg', fps=15, dpi=200)
        print(f" Animation saved successfully as {save_filename}")
    except Exception as e:
        print(f"Error saving as MP4: {e}. Animation will be shown instead.")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()