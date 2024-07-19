
import numpy as np


def EKF_localization(mu_prev, Sigma_prev, u_t, z_t, m):
    # Define matricies
    G_t = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    # Define matricies
    A_t = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    dt = 0.1
    B_t = dt * np.array([[np.cos(mu_prev[2]), 0],
                         [np.sin(mu_prev[2]), 1],
                         [0, 1]])
    R_t = np.array([[0.075 * np.cos(mu_prev[2]), 0, 0],
                    [0, 0.075 * np.sin(mu_prev[2]), 0],
                    [0, 0, 0.2]])

    C_t = np.array([[0.02 * np.cos(mu_prev[2]), 0, 0],
                    [0, 0.02 * np.cos(mu_prev[2]), 0],
                    [0, 0.0, 0.02]])

    Q_t = np.array([[0.02 * np.cos(mu_prev[2]), 0, 0],
                    [0, 0.02 * np.cos(mu_prev[2]), 0],
                    [0, 0.0, 0.02]])

    # Predict
    mu_t_bar = (A_t @ mu_prev + B_t @ u_t).reshape((3, 1))
    Sigma_t_bar = A_t @ Sigma_prev @ A_t.T + R_t

    # Update
    K_t = Sigma_t_bar @ (C_t.T @ np.linalg.inv(C_t @
                         Sigma_t_bar @ C_t.T + Q_t))

    z_t = np.array([[z_t[0]], [z_t[1]], [z_t[2]]])
    u_t.reshape(2, 1)
    mu_t = mu_t_bar + K_t @ (z_t - C_t @ mu_t_bar)

    kt_ct = K_t @ C_t
    Sigma_t = (np.eye(kt_ct.shape[0]) - kt_ct) @ Sigma_t_bar

    return mu_t, Sigma_t
