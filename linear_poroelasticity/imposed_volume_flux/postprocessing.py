"""This file collects all the data from the different regions and converts the
scaled variables back into the original x-coordinate, before creating a joint plot.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from quantity import Quantity

from fenics import *
delta_t = 0.01
N_x = 40
N_time = 200
b_M_num = 3.0
Pe_num = 2.0

v_f0_before = np.zeros(N_x + 1)
v_f0_data_bl = (pd.read_csv(f"data_bl/Q_step/vf_bM_{b_M_num}.csv", index_col=0)
                .to_numpy())
v_f0_after = np.ones(N_x + 1)
c_0_data = (pd.read_csv(f"data_no_bl/Q_step/c_bM_{b_M_num}.csv", index_col=0)
            .to_numpy())
M_0_data = (pd.read_csv(f"data_no_bl/Q_step/M_bM_{b_M_num}.csv", index_col=0)
            .to_numpy())
sigma_xx0_data = (pd.read_csv(f"data_no_bl/Q_step/sigma_bM_{b_M_num}.csv", index_col=0)
                  .to_numpy())
sigma_xx0_data_bl = (pd.read_csv(f"data_bl/Q_step/sigma_bM_{b_M_num}.csv", index_col=0)
                     .to_numpy())
phi_f1_data = (pd.read_csv(f"data_no_bl/Q_step/phi_bM_{b_M_num}.csv", index_col=0)
               .to_numpy())
phi_f1_data_bl = (pd.read_csv(f"data_bl/Q_step/phi_bM_{b_M_num}.csv", index_col=0)
                  .to_numpy())
u_s0_data = (pd.read_csv(f"data_no_bl/Q_step/us_bM_{b_M_num}.csv", index_col=0)
             .to_numpy())
u_s0_data_bl = (pd.read_csv(f"data_bl/Q_step/us_bM_{b_M_num}.csv", index_col=0)
                .to_numpy())

inside_bl = [v_f0_data_bl, sigma_xx0_data_bl, phi_f1_data_bl, u_s0_data_bl]
outside_bl = [c_0_data, M_0_data, sigma_xx0_data, phi_f1_data, u_s0_data]

# Set up our quantities (with plotting functionality)
mesh = IntervalMesh(N_x, 0, 1)
v_f0 = Quantity("$v_{f,0}$", "PuRd", 0, mesh)
c_0 = Quantity("$c_{0}$", "Reds", 1, mesh)
M_0 = Quantity("$\\mathcal{M}_{0}$", "Purples", 2, mesh)
sigma_xx0 = Quantity("$\\sigma_{xx,0}'$", "YlOrBr", 3, mesh)
phi_f1 = Quantity("$\\phi_{f,1}$", "Blues", 4, mesh)
u_s0 = Quantity("$u_{s,0}$", "Greens", 5, mesh)
quantities = [v_f0, c_0, M_0, sigma_xx0, phi_f1, u_s0]
quantities_bl = [v_f0, sigma_xx0, phi_f1, u_s0]

# Setting up the figure and axes
fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(4, 60/3), sharex=True)
Quantity.set_axs(quantities, axs)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)
times = np.linspace(0, N_time * delta_t, N_time + 1)

# Set up data saving
saving = [True, True, True, True, True, True]
short_quants = ["vf", "c", "M", "sigma", "phi", "u"]
file_names = [f"all_data/Q_step/{q}_bM_{b_M_num}.csv" for q in short_quants]
for file_name in file_names:
    pd.DataFrame().to_csv(file_name)

# Plot for each timestep before t = 1
print("Before the boundary layer")
v_f0.f = v_f0_before
v_f0.plot(norm, 0.0, True, file_names[0])
for i in range(N_time // 2):
    print(f"t = {i * delta_t}")
    for j in range(len(quantities) - 1):
        quantities[j + 1].f = outside_bl[j][:, i]
    Quantity.plot_quantities(quantities[1:], norm, i * delta_t, saving[1:], file_names[1:])

# Plot for each timestep inside the boundary layer
print("Inside the boundary layer")
for i in range(N_time):
    print(f"T = {i * delta_t}")
    for j in range(len(quantities_bl)):
        quantities_bl[j].f = inside_bl[j][:, i]
    t = 1.0 + i * delta_t ** 3
    Quantity.plot_quantities(quantities_bl, norm, t,
                             [saving[0]] + saving[3:], [file_names[0]] + file_names[3:],
                             alpha=0.5)

# Plot for each timestep after t = 1
print("After the boundary layer")
for i in range(N_time // 2, N_time):
    print(f"t = {i * delta_t}")
    for j in range(len(quantities) - 1):
        quantities[j + 1].f = outside_bl[j][:, i]
    Quantity.plot_quantities(quantities[1:], norm, i * delta_t, saving[1:], file_names[1:])
v_f0.f = v_f0_after
v_f0.plot(norm, N_time * delta_t, True, file_names[0])

# Create the legends and save the figure

# v_f,0
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=v_f0.cmap),
             orientation='vertical',
             label='$t$', ax=v_f0.ax)
v_f0.label_plot(title="Leading order fluid velocity")

# c_0
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=c_0.cmap),
             orientation='vertical',
             label='$t$', ax=c_0.ax)
c_0.label_plot(title="Leading order solute concentration")

# M_0
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=M_0.cmap),
             orientation='vertical',
             label='$t$', ax=M_0.ax)
M_0.label_plot(title="Leading order p-wave modulus")

# sigma_xx,0'
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=sigma_xx0.cmap),
             orientation='vertical',
             label='$t$', ax=sigma_xx0.ax)
sigma_xx0.label_plot(title="Leading order Terzaghi stress")

# phi_f,1
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f1.cmap),
             orientation='vertical',
             label='$t$', ax=phi_f1.ax)
phi_f1.label_plot(title="First order porosity")

# u_s,0
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s0.cmap),
             orientation='vertical',
             label='$t$', ax=u_s0.ax)
u_s0.label_plot(title="Leading order displacement")

# Save figure
fig.suptitle(f"FEniCS solution with $b_M={b_M_num}$")
fig.savefig(f"plots/Q_step_total/fenics_b_M_{b_M_num}_Pe_{Pe_num}.png", bbox_inches="tight")
