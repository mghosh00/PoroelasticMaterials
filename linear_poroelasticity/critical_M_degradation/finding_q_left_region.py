import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

M_0_df = (pd.read_csv("x_data/c0_diff_no_flux/M_bM_1.0.csv", index_col=0)
          )
M_0_array = M_0_df.to_numpy()
# The numpy solution for x_c(t) and t_c
M_c_num, N_x, delta_t, N_time = 0.7, 40, 0.01, 200
x_c = np.argmax(np.where(M_0_array < M_c_num, M_0_array, 0.0), axis=0) / N_x
i_c, t_c = np.argmax(x_c), np.argmax(x_c) * delta_t
# Now we convert to new time coordinates tau = t - t_c
x_c = x_c[i_c - 1:]
x_c[0] = 1.0
x_c_dot = np.gradient(x_c) / delta_t
N_time = N_time - i_c + 1

# To avoid later computational errors (should not make a different to the calculations)
x_c[0] = 0.999
x_c_dot += 1e-7

u_s0_xi_array = (pd.read_csv("xi_L_data/c0_diff_no_flux/u_bM_1.0.csv", index_col=0)
                 ).to_numpy()[:, :-1]
sigma_xx0_xi_array = (pd.read_csv("xi_L_data/c0_diff_no_flux/sigma_bM_1.0.csv", index_col=0)
                      ).to_numpy()[:, :-1]

dxi = 1 / 40
dt = 1 / 100
dus0_dt = np.gradient(u_s0_xi_array, axis=1) / dt
dsigma_xx0_dxi = np.gradient(sigma_xx0_xi_array, axis=0) / dxi
dus0_dxi = np.gradient(u_s0_xi_array, axis=0) / dxi

xarr = np.linspace(0, 1, 41)

norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)
whole_map = mpl.colormaps['YlGn']
cmap = mpl.colors.LinearSegmentedColormap.from_list("YlGn_subset",
                                                    whole_map(np.linspace(0.3, 1.0, 100)))
fig, ax = plt.subplots(1, 1)
for t in range(sigma_xx0_xi_array.shape[1]):
    ax.plot(xarr, dus0_dt[:, t] - dsigma_xx0_dxi[:, t] / x_c[t] -
            xarr * x_c_dot[t] / x_c[t] * dus0_dxi[:, t], color=cmap(norm(t * delta_t)))
ax.set_xlabel("$\\xi_{L}$", fontsize='xx-large')
ax.set_ylabel("$q_{0}$", fontsize='xx-large')
ax.set_yscale('symlog')
# ax.set_title("Leading-order volume flux")
cb_q = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    orientation='vertical',
                    ax=ax)
cb_q.set_label(label=r'$\tau$', fontsize='xx-large')
# plt.ylim(-5, 0)
fig.savefig("plots/new_q0/q0.png", bbox_inches='tight')
