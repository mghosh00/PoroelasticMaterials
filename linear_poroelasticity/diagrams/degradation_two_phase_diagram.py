import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

t_c = 1.0
x = np.linspace(0, 1, 100)
t = t_c - np.log(x)

plt.plot(x, t, '-k', lw=2, label=r'$x_c(t)$')
plt.axvline(1.0, ls='-', color='black', lw=1.0)
plt.axhline(t_c, ls='--', color='red', lw=1.5, label=r'$t_c$')
plt.xlabel(r'$x$', fontsize='xx-large')
plt.ylabel(r'$t$', fontsize='xx-large')
plt.ylim(0.0, 5.0)
plt.xlim(0.0, 1.0)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelleft=False)
plt.legend(fontsize='xx-large')
# plt.title("$x$-$t$ diagram for the degrading poroelastic material")
plt.text(0.35, 0.4, r"poroelastic material", fontsize='x-large')
plt.text(0.1, 1.5, r"poroelastic material", fontsize='x-large')
plt.text(0.45, 3.0, r"two-phase flow", fontsize='x-large')
plt.savefig('degradation_two_phase_diagram.png', bbox_inches='tight')
