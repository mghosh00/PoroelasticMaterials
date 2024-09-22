import numpy as np
import matplotlib.pyplot as plt

t_c = 1.0
x = np.linspace(0, 1, 100)
t = t_c - np.log(x)

plt.plot(x, t, '-k', lw=2, label='$x_c(t)$')
plt.axvline(1.0, ls='-', color='black', lw=1.0)
plt.axhline(t_c, ls='--', color='red', lw=1.5, label='$t_c$')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.ylim(0.0, 5.0)
plt.xlim(0.0, 1.0)
plt.legend()
plt.title("$x$-$t$ diagram for the degrading poroelastic material")
plt.savefig('degradation_two_phase_diagram.png')
