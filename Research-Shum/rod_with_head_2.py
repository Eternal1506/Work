# kirchhoff_rod_with_head.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
import numba
import time
import os

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
    "b_amp": 0.8,
    "sigma_freq": 275,
    "animation_interval": 40,
    "animation_steps_skip": 100,
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
        return np.array([self.D1[k], self.D2[k], self.D3[k]])

    def update_intrinsic_curvature(self):
        if self.p["scenario"] == "flagellar_wave":
            k = self.p["k_wave"]
            b = self.p["b_amp"]
            sigma = self.p["sigma_freq"]
            self.Omega[:, 0] = -k ** 2 * b * np.sin(k * self.s_vals - sigma * self.time)
            self.Omega[:, 1] = 0.0
            self.Omega[:, 2] = 0.0

    def compute_internal_forces_and_moments(self):
        F_half = np.zeros((self.M - 1, 3))
        N_half = np.zeros((self.M - 1, 3))
        self.D_half_matrices = []
        for k in range(self.M - 1):
            Dk_mat = self._get_D_matrix(k)
            Dkp1_mat = self._get_D_matrix(k + 1)
            Ak = Dkp1_mat @ Dk_mat.T
            sqrt_Ak = get_rotation_matrix_sqrt(Ak)
            D_half_k_mat = sqrt_Ak @ Dk_mat
            self.D_half_matrices.append(D_half_k_mat)
            D1_h, D2_h, D3_h = D_half_k_mat[0], D_half_k_mat[1], D_half_k_mat[2]
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
            N_coeffs = np.array([
                self.a[0] * (np.dot(dD2ds, D3_h) - self.Omega[k, 0]),
                self.a[1] * (np.dot(dD3ds, D1_h) - self.Omega[k, 1]),
                self.a[2] * (np.dot(dD1ds, D2_h) - self.Omega[k, 2])
            ])
            N_half[k] = N_coeffs[0] * D1_h + N_coeffs[1] * D2_h + N_coeffs[2] * D3_h
        return F_half, N_half

    def compute_fluid_forces_on_rod(self, F_half, N_half):
        f_on_rod = np.zeros((self.M, 3))
        n_on_rod = np.zeros((self.M, 3))
        F_b_start = np.zeros(3)
        F_b_end = np.zeros(3)
        N_b_start = np.zeros(3)
        N_b_end = np.zeros(3)
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

        # If soft attachment: add spring force at base to enforce soft constraint
        if self.has_head and self.use_soft_attachment:
            attach_world = self.X_head + self.E_head @ self.r_attach_head
            delta = self.X[0] - attach_world
            f_spring = -self.attachment_k_spring * delta  # force applied on rod base
            # distribute spring force to f_on_rod[0] (force density) by adding f_spring/ds
            f_on_rod[0] += f_spring / self.ds

            # torque from spring: zero here unless you want to add moment
        return f_on_rod, n_on_rod

    def compute_head_force_and_torque(self, f_on_rod, n_on_rod):
        """
        Compute the single point head force g0_head and torque m0_head that balance filament forces and torques.
        Note: we return quantities in units of force and torque already multiplied by ds (consistent with g0_all usage).
        """
        # total filament force (point forces) = sum(f_on_rod * ds)
        total_force = np.sum(f_on_rod * self.ds, axis=0)
        # total torque about the head centre:
        rel = self.X - self.X_head  # (M,3)
        cross_terms = np.cross(rel, f_on_rod * self.ds)
        total_torque = np.sum(n_on_rod * self.ds, axis=0) + np.sum(cross_terms, axis=0)
        # head supplies negative
        g0_head = -total_force
        m0_head = -total_torque
        return g0_head, m0_head

    def compute_velocities(self, f_on_rod, n_on_rod):
        """
        Build arrays including head and call the numba core which accepts per-source epsilons.
        Returns u_rod, w_rod, u_head, w_head
        """
        # filament point forces (already densities) -> point forces by multiplying ds
        g0_fil = f_on_rod * self.ds
        m0_fil = n_on_rod * self.ds

        if not self.has_head:
            # no head: call with M sources = M
            eps_vec = np.full(self.M, self.epsilon_reg)
            u_all, w_all = _compute_velocities_core(self.M, self.X, g0_fil, m0_fil, eps_vec, self.mu)
            return u_all, w_all, None, None
        else:
            g0_head, m0_head = self.compute_head_force_and_torque(f_on_rod, n_on_rod)
            X_all = np.vstack([self.X, self.X_head.reshape(1, 3)])
            g0_all = np.vstack([g0_fil, g0_head.reshape(1, 3)])
            m0_all = np.vstack([m0_fil, m0_head.reshape(1, 3)])
            eps_vec = np.empty(self.M + 1)
            for i in range(self.M):
                eps_vec[i] = self.epsilon_reg
            eps_vec[self.M] = self.eps_head
            Mtot = self.M + 1
            u_all, w_all = _compute_velocities_core(Mtot, X_all, g0_all, m0_all, eps_vec, self.mu)
            u_rod = u_all[:self.M, :]
            w_rod = w_all[:self.M, :]
            u_head = u_all[self.M, :]
            w_head = w_all[self.M, :]
            return u_rod, w_rod, u_head, w_head

    def update_state(self, u_rod, w_rod):
        self.X += u_rod * self.dt
        for k in range(self.M):
            wk = w_rod[k]
            wk_mag = np.linalg.norm(wk)
            if wk_mag > 1e-12:
                axis_e = wk / wk_mag
                angle_theta = wk_mag * self.dt
                R_matrix = get_rodrigues_rotation_matrix(axis_e, angle_theta)
                # rotate director frames (world->world rotation)
                self.D1[k] = (R_matrix @ self.D1[k])
                self.D2[k] = (R_matrix @ self.D2[k])
                self.D3[k] = (R_matrix @ self.D3[k])

    def update_head_pose(self, u_head, w_head):
        if not self.has_head:
            return
        # translate
        self.X_head += u_head * self.dt
        wmag = np.linalg.norm(w_head)
        if wmag > 1e-12:
            axis = w_head / wmag
            theta = wmag * self.dt
            R = get_rodrigues_rotation_matrix(axis, theta)
            # update E_head: world <- R @ world (rotate body axes)
            self.E_head = R @ self.E_head
        # enforce attachment
        if self.enforce_attachment and not self.use_soft_attachment:
            attach_world = self.X_head + self.E_head @ self.r_attach_head
            self.X[0] = attach_world.copy()
            # optionally align directors at base with head frame
            self.D3[0] = self.E_head[:, 2].copy()
            self.D1[0] = self.E_head[:, 0].copy()
            self.D2[0] = self.E_head[:, 1].copy()

    def simulation_step(self):
        self.update_intrinsic_curvature()
        F_half, N_half = self.compute_internal_forces_and_moments()
        f_on_rod, n_on_rod = self.compute_fluid_forces_on_rod(F_half, N_half)
        u_rod, w_rod, u_head, w_head = self.compute_velocities(f_on_rod, n_on_rod)

        # update rod and head states
        self.update_state(u_rod, w_rod)
        if self.has_head and u_head is not None:
            self.update_head_pose(u_head, w_head)

        self.time += self.dt
        return u_rod, w_rod, f_on_rod, n_on_rod

    # --- flow field evaluation wrapper using numba core ---
    def evaluate_velocity_field_on_grid(self, eval_points):
        """
        eval_points: (Ne,3)
        returns u_eval (Ne,3)
        """
        # build sources as in compute_velocities
        # need to compute f_on_rod, n_on_rod for current state to build g0 and m0 - caller should supply these
        # To allow plotting at arbitrary times, the function will accept eval_points and compute contributions
        # using the most recent f/n if desired. Here we'll return zeros if fluid forces are not known.
        # For convenience of using it after a step, the user should call compute_fluid_forces_on_rod prior and keep f_on_rod/n_on_rod.
        raise NotImplementedError("Use standalone evaluate_velocity_field(...) provided in module which accepts g0,m0,X_all.")

# --- Standalone evaluate_velocity_field function using numba core ---
def evaluate_velocity_field(eval_points, X_sources, g0_sources, m0_sources, eps_sources, mu):
    """
    eval_points: (Ne,3) numpy array
    X_sources: (Ns,3)
    g0_sources, m0_sources: (Ns,3)
    eps_sources: (Ns,) array of epsilons
    returns u_eval (Ne,3)
    """
    # ensure contiguous arrays of right dtype for numba
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
    history_X = []
    history_u = []
    history_w = []
    history_f = []
    history_n = []
    history_X_head = []
    print(f"Starting simulation for {PARAMS['total_time']}s, with dt={PARAMS['dt']}s ({num_steps} steps).")
    print(f"Rod initial shape: {PARAMS.get('initial_shape', 'straight')}")
    print(f"Effective rod length: {rod.L_eff:.4f} um")
    print(f"Regularization epsilon (filament): {rod.epsilon_reg:.4f} um")
    if rod.has_head:
        print(f"Head radius (epsilon_head): {rod.eps_head:.4f} um")
    start_time = time.time()

    # To speed up first call JIT, do a single small evaluate call (warm-up)
    # Warm up the numba functions once (cheap)
    try:
        # create tiny dummy arrays
        Xs_dummy = np.zeros((1,3))
        g_dummy = np.zeros((1,3))
        m_dummy = np.zeros((1,3))
        eps_dummy = np.array([rod.epsilon_reg])
        pts_dummy = np.zeros((1,3))
        _compute_velocities_core(1, Xs_dummy, g_dummy, m_dummy, eps_dummy, rod.mu)
        _evaluate_velocity_field_core(pts_dummy, Xs_dummy, g_dummy, m_dummy, eps_dummy, rod.mu)
    except Exception:
        pass

    for step in range(num_steps):
        u_rod, w_rod, f_on_rod, n_on_rod = rod.simulation_step()

        if step % PARAMS["animation_steps_skip"] == 0:
            history_X.append(rod.X.copy())
            history_u.append(u_rod.copy())
            history_w.append(w_rod.copy())
            history_f.append(f_on_rod.copy())
            history_n.append(n_on_rod.copy())
            if rod.has_head:
                history_X_head.append(rod.X_head.copy())

            if step % (PARAMS["animation_steps_skip"] * 10) == 0:
                current_sim_time = time.time()
                elapsed_wall_time = current_sim_time - start_time
                print(f"Step {step}/{num_steps}, Sim Time: {step*PARAMS['dt']:.2e} s, Wall Time: {elapsed_wall_time:.2f}s. Max X: {np.max(np.abs(rod.X)):.2f}")

            if np.max(np.abs(rod.X)) > rod.L_eff * 20:
                print("Simulation unstable. Coordinates exploded.")
                break
            if np.isnan(rod.X).any() or np.isinf(rod.X).any():
                print("NaN/Inf detected in coordinates. Simulation unstable.")
                break
            if np.isnan(rod.D1).any() or np.isnan(rod.D2).any() or np.isnan(rod.D3).any():
                print("NaN detected in directors. Simulation unstable.")
                break

        # Optional debug plotting
        if step % PARAMS["debug_plot_interval_steps"] == 0 and PARAMS["debugging"]:
            print(f"Debug plot at time {rod.time:.6e}")
            plt.figure(); plt.plot(rod.s_vals, rod.Omega[:,0], 'o-'); plt.xlabel('s'); plt.ylabel('Omega1'); plt.show()

    end_time = time.time()
    print(f"Simulation finished. Total wall time: {end_time - start_time:.2f} seconds.")

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
                reshaped_data = np.array(data).reshape(num_frames * M, 3)
                np.savetxt(path, reshaped_data, fmt='%.8e', header=header, comments='')
                print(f" -> Saved: {path}")

            save_history('history_X.txt', history_X, 'x y z')
            save_history('history_u.txt', history_u, 'ux uy uz')
            save_history('history_w.txt', history_w, 'wx wy wz')
            save_history('history_f.txt', history_f, 'fx fy fz')
            save_history('history_n.txt', history_n, 'nx ny nz')

            with open(os.path.join(output_dir, 'simulation_params.txt'), 'w') as f:
                f.write(f"num_frames = {num_frames}\n")
                f.write(f"M = {M}\n")
                f.write(f"dt_snapshot = {PARAMS['animation_steps_skip'] * PARAMS['dt']}\n")
                for key, value in PARAMS.items():
                    f.write(f"{key} = {value}\n")
            print(f" -> Saved: {os.path.join(output_dir, 'simulation_params.txt')}")
            print("\nData saved successfully.")
            print("To load data: loaded_X = np.loadtxt('simulation_history/history_X.txt', skiprows=1).reshape(num_frames, M, 3)")
        except Exception as e:
            print(f"Error saving history: {e}")

    # --- Animation of rod + optional flow field (2D slice) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=3)
    head_point, = ax.plot([], [], [], 'o', color='red', markersize=6)
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    if history_X:
        all_coords = np.concatenate(history_X, axis=0)
        x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        z_min, z_max = np.min(all_coords[:, 2]), np.max(all_coords[:, 2])
        center_x, center_y, center_z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        plot_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min, rod.L_eff * 0.5]) * 0.6
        if plot_range < 1e-6:
            plot_range = rod.L_eff if rod.L_eff > 0 else 1.0
        ax.set_xlim([center_x - plot_range, center_x + plot_range])
        ax.set_ylim([center_y - plot_range, center_y + plot_range])
        ax.set_zlim([center_z - plot_range, center_z + plot_range])
        try:
            ax.set_aspect('equal', adjustable='box')
        except NotImplementedError:
            pass
    else:
        lim = rod.L_eff / 2 if rod.L_eff > 0 else 1
        ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim / 2, lim / 2])

    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)"); ax.set_zlabel("Z (um)")
    ax.set_title(f"Kirchhoff Rod Dynamics (Initial: {PARAMS.get('initial_shape', 'straight')})")
    ax.view_init(elev=20., azim=-35)

    def init_animation():
        line.set_data([], [])
        line.set_3d_properties([])
        head_point.set_data([], [])
        head_point.set_3d_properties([])
        time_text.set_text('')
        return line, head_point, time_text

    def update_animation(frame_idx):
        X_data = history_X[frame_idx]
        line.set_data(X_data[:, 0], X_data[:, 1])
        line.set_3d_properties(X_data[:, 2])
        if rod.has_head and history_X_head:
            Xh = history_X_head[frame_idx]
            head_point.set_data([Xh[0]], [Xh[1]]); head_point.set_3d_properties([Xh[2]])
        current_time = frame_idx * PARAMS["animation_steps_skip"] * PARAMS["dt"]
        time_text.set_text(f'Time: {current_time:.3e} s')
        return line, head_point, time_text

    if history_X:
        ani = FuncAnimation(fig, update_animation, frames=len(history_X),
                            init_func=init_animation, blit=False, interval=PARAMS["animation_interval"])
        try:
            save_filename = 'kirchhoff_rod_with_head_animation.mp4'
            print(f"Attempting to save animation as {save_filename}...")
            ani.save(save_filename, writer='ffmpeg', fps=15, dpi=150)
            print(f"Animation saved as {save_filename}")
        except Exception as e:
            print(f"Error saving as MP4: {e}. Trying GIF...")
        try:
            ani.save('kirchhoff_rod_with_head_animation.gif', writer='pillow', fps=10, dpi=100)
            print("Animation saved as kirchhoff_rod_with_head_animation.gif")
        except Exception as e:
            print(f"Error saving as GIF: {e}")

        plt.tight_layout(); plt.show()
    else:
        print("No history data to animate.")

    # --- Example: compute + plot a 2D flow slice at final frame if saved history exists ---
    if history_X:
        # compute g0 and m0 for final state to evaluate flow
        final_f = history_f[-1]
        final_n = history_n[-1]
        # filament point forces and torques -> multiply by ds to get point-force/torque
        g0_fil = final_f * rod.ds
        m0_fil = final_n * rod.ds
        if rod.has_head:
            g0_head, m0_head = rod.compute_head_force_and_torque(final_f, final_n)
            X_all = np.vstack([rod.X, rod.X_head.reshape(1, 3)])
            g0_all = np.vstack([g0_fil, g0_head.reshape(1, 3)])
            m0_all = np.vstack([m0_fil, m0_head.reshape(1, 3)])
            eps_all = np.empty(rod.M + 1)
            for i in range(rod.M):
                eps_all[i] = rod.epsilon_reg
            eps_all[rod.M] = rod.eps_head
        else:
            X_all = rod.X.copy()
            g0_all = g0_fil.copy()
            m0_all = m0_fil.copy()
            eps_all = np.full(rod.M, rod.epsilon_reg)

        # build a 2D grid in the plane z = center_z (or z=0)
        nx, ny = 40, 40
        xlim = (-2.0, 2.0)
        ylim = (-2.0, 2.0)
        xs = np.linspace(xlim[0], xlim[1], nx)
        ys = np.linspace(ylim[0], ylim[1], ny)
        Xg, Yg = np.meshgrid(xs, ys)
        pts = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, 0.0)])
        print("Evaluating velocity field on 2D slice (num points = {})...".format(pts.shape[0]))
        us = evaluate_velocity_field(pts, X_all, g0_all, m0_all, eps_all, rod.mu)
        Ux = us[:, 0].reshape(Xg.shape); Uy = us[:, 1].reshape(Xg.shape)
        speed = np.sqrt(Ux ** 2 + Uy ** 2)

        plt.figure(figsize=(7, 6))
        plt.streamplot(xs, ys, Ux, Uy, density=1.5)
        plt.scatter(X_all[:, 0], X_all[:, 1], c='k', s=8)
        # color background by speed (showing imshow under streamplot)
        plt.imshow(speed, origin='lower', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), alpha=0.6)
        plt.colorbar(label='speed (um/s)')
        plt.title('Flow slice (z=0)')
        plt.xlabel('x'); plt.ylabel('y')
        plt.tight_layout()
        plt.show()
