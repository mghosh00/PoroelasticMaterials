"""This file collects all the data from the different regions and converts the
scaled variables back into the original x-coordinate, before creating a joint plot.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from quantity import Quantity
from fenics import *
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

M_c_num = 0.7
delta_t = 0.01
N_x = 40
N_time = 200
b_M_num = 1.0
D_m_num = 0.5


M_0_early = (pd.read_csv(f"x_data/c0_diff_no_flux/M_bM_{b_M_num}.csv", index_col=0)
             .to_numpy())
c_0_early = (pd.read_csv(f"x_data/c0_diff_no_flux/c_bM_{b_M_num}.csv", index_col=0)
             .to_numpy())
sigma_early = (pd.read_csv(f"x_data/c0_diff_no_flux/sigma_bM_{b_M_num}.csv", index_col=0)
               .to_numpy())
phi_early = (pd.read_csv(f"x_data/c0_diff_no_flux/phi_bM_{b_M_num}.csv", index_col=0)
             .to_numpy())
u_early = (pd.read_csv(f"x_data/c0_diff_no_flux/u_bM_{b_M_num}.csv", index_col=0)
           .to_numpy())
early_arrays = [c_0_early, M_0_early, sigma_early, phi_early, u_early]
sigma_late_L_xi = (pd.read_csv(f"xi_L_data/new_q0/sigma_bM_{b_M_num}.csv", index_col=0)
                   .to_numpy()[:, :-1])
phi_late_L_xi = (pd.read_csv(f"xi_L_data/new_q0/phi_bM_{b_M_num}.csv", index_col=0)
                 .to_numpy()[:, :-1])
u_late_L_xi = (pd.read_csv(f"xi_L_data/new_q0/u_bM_{b_M_num}.csv", index_col=0)
               .to_numpy()[:, :-1])
phi_late_R_xi = (pd.read_csv(f"xi_R_data/new_q0/phi_bM_{b_M_num}.csv", index_col=0)
                 .to_numpy()[:, :-1])
v_s0_late_R_xi = (pd.read_csv(f"xi_R_data/new_q0/vs_bM_{b_M_num}.csv", index_col=0)
                  .to_numpy()[:, :-1])
v_f0_late_R_xi = (pd.read_csv(f"xi_R_data/new_q0/vf_bM_{b_M_num}.csv", index_col=0)
                  .to_numpy()[:, :-1])

# The numpy solution for x_c(t) and t_c
x_c = np.argmax(np.where(M_0_early < M_c_num, M_0_early, 0.0), axis=0) / N_x
i_c, t_c = np.argmax(x_c), np.argmax(x_c) * delta_t
# Now we convert to new time coordinates tau = t - t_c
x_c = x_c[i_c - 1:]
x_c[0] = 1.0


def xi_tau_to_x_t(f: np.array, x_c: np.array, left: bool):
    """Change the quantity from (xi, tau) coordinates to (x, t) where
    \\xi_{L} = x / x_{c}(t), \\xi_{R} = (x - x_{c}(t)) / (1 - x_{c}(t)),
    \\tau = t - t_{c}. The function does different things depending on
    whether this is for xi_L or xi_R. We also fit onto the new mesh, which
    will involve some interpolation.

    :param f: Some quantity in (xi, tau) coordinates (np array).
    :param x_c: The boundary curve x_{c}(t).
    :param left: Whether we are on the left or right.
    :return: The new array in (x, t) coordinates.
    """
    f_new = []

    # Now focus on each timestep
    for i in range(f.shape[1]):
        # Find the specific timestep
        f_i = f[:, i]
        # How many points there are for left and right regions
        N_part = int(x_c[i] * N_x) if left else N_x - int(x_c[i] * N_x)
        # Shrink region to [0, 1] (using transformation) and interpolate onto
        # only the left part of the grid
        f_i_part = np.interp(np.linspace(0, 1, N_part + 1),
                             np.linspace(0, 1, N_x + 1),
                             f_i)
        # Fill the left or right of array with NaNs depending on which region we are in
        if left:
            f_i_part = np.concatenate([f_i_part, np.full(N_x - N_part, np.nan)])
        else:
            f_i_part = np.concatenate([np.full(N_x - N_part, np.nan), f_i_part])

        f_new.append(f_i_part)
    return np.array(f_new).transpose()


# Convert our quantities from (xi, tau) to (x, t) coords
sigma_late_L = xi_tau_to_x_t(sigma_late_L_xi, x_c, left=True)
phi_late_L = xi_tau_to_x_t(phi_late_L_xi, x_c, left=True)
u_late_L = xi_tau_to_x_t(u_late_L_xi, x_c, left=True)
phi_late_R = xi_tau_to_x_t(phi_late_R_xi, x_c, left=False)
vs_late_R = xi_tau_to_x_t(v_s0_late_R_xi, x_c, left=False)
vf_late_R = xi_tau_to_x_t(v_f0_late_R_xi, x_c, left=False)

# Set up our quantities (with plotting functionality)
mesh = IntervalMesh(N_x, 0, 1)
c_0 = Quantity("$c_{0}$", "Reds", 0, mesh)
M_0 = Quantity("$\\mathcal{M}_{0}$", "Purples", 1, mesh)
sigma_xx0 = Quantity("$\\sigma_{xx,0}'$", "YlOrBr", 2, mesh)
phi_f1 = Quantity("$\\phi_{f,1}$", "Blues", 3, mesh)
u_s0 = Quantity("$u_{s,0}$", "Greens", 4, mesh)
v_s0 = Quantity("$v_{s,0}$", "Oranges", 5, mesh)
v_f0 = Quantity("$v_{f,0}$", "PuRd", 6, mesh)
quantities = [c_0, M_0, sigma_xx0, phi_f1, u_s0, v_s0, v_f0]

# Setting up the figure and axes
# fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8, 50/3), sharex=True)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 10/3), sharex=True)
# fig, axs_before_tc = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
# fig, axs_after_tc = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
# axs_before_tc, axs_after_tc = [axs[i, 0] for i in range(5)], [axs[i, 1] for i in range(5)]
axs_before_tc, axs_after_tc = axs[0], axs[1]
# Quantity.set_axs(quantities[:5], axs_before_tc)
Quantity.set_axs([quantities[3]], [axs_before_tc])
norm_before_tc = mpl.colors.Normalize(vmin=0.0, vmax=t_c)
norm_after_tc = mpl.colors.Normalize(vmin=t_c, vmax=N_time * delta_t)
times = np.linspace(0, N_time * delta_t, N_time + 1)

# Set up data saving
saving = [True, True, True, True, True, True, True]
short_quants = ["c", "M", "sigma", "phi", "u", "vs", "vf"]
saving = saving[:5]
short_quants = short_quants[:5]
file_names = [f"all_data/new_q0/{q}_bM_{b_M_num}.csv" for q in short_quants]
for file_name in file_names:
    pd.DataFrame().to_csv(file_name)

# Mark M_c
# M_0.ax.axhline(0.7, color='black', linestyle='--', label=r'$\mathcal{M}_{c}$')
# M_0.ax.legend()

# Plot for each time step before the region splits
for i in range(i_c):
    print(i * delta_t)
    for j in range(len(quantities) - 2):
        quantities[j].f = early_arrays[j][:, i]
    # Quantity.plot_quantities(quantities[:5], norm_before_tc, i * delta_t, saving[:5], file_names[:5])
    Quantity.plot_quantities([quantities[3]], norm_before_tc, i * delta_t, [saving[3]], [file_names[3]])


def colorbar_and_labels(_fig, _norm, _quantity, _t_label, _fontsize=None,
                        _title='', _xlabel='', _label_size=None):
    cb = _fig.colorbar(mpl.cm.ScalarMappable(norm=_norm, cmap=_quantity.cmap),
                       orientation='vertical', ax=_quantity.ax)
    cb.set_label(label=_t_label, fontsize=_fontsize)
    _quantity.label_plot(title=_title, x_label=_xlabel, label_size=_label_size)


# Create the legends
# colorbar_and_labels(fig, norm_before_tc, c_0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large')
# colorbar_and_labels(fig, norm_before_tc, M_0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large')
# colorbar_and_labels(fig, norm_before_tc, sigma_xx0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large')
colorbar_and_labels(fig, norm_before_tc, phi_f1, r'$t$', _fontsize='xx-large',
                    _label_size='xx-large', _xlabel=r'$x$')
# colorbar_and_labels(fig, norm_before_tc, u_s0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large', _xlabel=r'$x$')
# c_0.ax.set_ylim(-0.05, 1.05)
# M_0.ax.set_ylim(0.69, 1.01)
# sigma_xx0.ax.set_ylim(-1.05, 0.05)
phi_f1.ax.set_ylim(-0.8, 0.85)
# u_s0.ax.set_ylim(-0.05, 0.95)
# fig.savefig(f"plots/new_q0/fenics_b_M_{b_M_num}_D_m_{D_m_num}_before_tc.png", bbox_inches="tight")


# Change the axes
# Quantity.set_axs(quantities[:5], axs_after_tc)
Quantity.set_axs([quantities[3]], [axs_after_tc])

# Mark M_c
# M_0.ax.axhline(0.7, color='black', linestyle='--', label=r'$\mathcal{M}_{c}$')
# M_0.ax.legend()

# Plot for each time step after the region splits
for i in range(i_c, N_time):
    print(i * delta_t)
    # Plot quantities which span the region
    c_0.f = c_0_early[:, i]
    M_0_early[M_0_early < M_c_num] = M_c_num
    M_0.f = M_0_early[:, i]
    # Plot the left-hand quantities
    # sigma_xx0.f = sigma_late_L[:, i - i_c]
    phi_f1.f = phi_late_L[:, i - i_c]
    u_s0.f = u_late_L[:, i - i_c]
    # Quantity.plot_quantities(quantities[:5], norm_after_tc, i * delta_t, saving[:5], file_names[:5])
    Quantity.plot_quantities([quantities[3]], norm_after_tc, i * delta_t, [saving[3]], [file_names[3]])
    # And plot the right-hand quantities
    phi_f1.f = phi_late_R[:, i - i_c]
    v_s0.f = vs_late_R[:, i - i_c]
    v_f0.f = vf_late_R[:, i - i_c]
    # Quantity.plot_quantities([phi_f1, v_s0, v_f0], norm, i * delta_t,
    #                          [True, True, True],
    #                          [file_names[3], file_names[5], file_names[6]])
    # Quantity.plot_quantities([phi_f1, v_s0], norm, i * delta_t,
    #                          [True, True],
    #                          [file_names[3], file_names[5]])
    Quantity.plot_quantities([phi_f1], norm_after_tc, i * delta_t,
                             [True],
                             [file_names[3]])

# Create the legends and save the figure
# colorbar_and_labels(fig, norm_after_tc, c_0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large')
# colorbar_and_labels(fig, norm_after_tc, M_0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large')
# colorbar_and_labels(fig, norm_after_tc, sigma_xx0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large')
colorbar_and_labels(fig, norm_after_tc, phi_f1, r'$t$', _fontsize='xx-large',
                    _label_size='xx-large', _xlabel=r'$x$')
# colorbar_and_labels(fig, norm_after_tc, u_s0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large', _xlabel=r'$x$')
# colorbar_and_labels(fig, norm, v_s0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large', _xlabel=r'$x$')
# colorbar_and_labels(fig, norm, v_f0, r'$t$', _fontsize='xx-large',
#                     _label_size='xx-large', _xlabel=r'$x$')
# c_0.ax.set_ylim(-0.05, 1.05)
# M_0.ax.set_ylim(0.69, 1.01)
# sigma_xx0.ax.set_ylim(-1.05, 0.05)
phi_f1.ax.set_ylim(-0.8, 0.85)
# u_s0.ax.set_ylim(-0.05, 0.95)

# Save figure
# fig.suptitle(f"FEniCS solution with $b_M={b_M_num}$")
fig.subplots_adjust(wspace=0.5)
fig.savefig(f"plots/new_q0/fenics_b_M_{b_M_num}_D_m_{D_m_num}_split_phi.png", bbox_inches="tight")
