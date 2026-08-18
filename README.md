# Modelling pore closure in weakening poroelastic media

In this repository, we use the finite element method to simulate a mathematical model for the solute-induced weakening of poroelastic media which can, in some cases, lead to localised pore closure. We also use a range of analytical techniques to interrogate the model and compare to our numerical predictions.

## Installation

Firstly, ensure that legacy FEniCS is installed on your device. See the installation guide from the FEniCS website [here](https://fenicsproject.org/download/archive/) for more details. Note that if you are using a Windows device, you will need to install FEniCS using WSL (Windows Subsystem for Linux). Next, clone this repository on your device using

```
git clone git@github.com:mghosh00/PoroelasticMaterials.git
```

and `cd` into the resulting directory. If you wish, create and activate a virtual environment. From here, install the package via

```
pip install -e .
```

If you would like to work outside a virtual environment and already have `numpy`, `scipy`, `matplotlib`, `pandas` and `Pillow` installed on your device, you can install the package using the following command instead:

```
pip install -e . --no-dependencies
```

## Setting up a simulation

If you wish to use an existing parameter set with which to simulate the model, skip to the section on **Running a simulation**. To create your own parameter set, first `cd` into `nonlinear_poroelasticity/weakening/resources` and create a new directory inside either `phys` or `nonphys` depending on whether your parameters are based on a physical problem or not. Find a `params.json` file from any of the example directories and paste this here. Edit the file as required and see the below table for a description of the parameter values.

|Category|Parameter|Description|Unit|
|--|--|----------|--|
|`comp`|`delta_tau`|Step size for time parameter `tau`. This parameter is linearly spaced.|None|
||`N_time`|Number of timepoints.|None|
||`N_x`|Number of spatial meshpoints.|None|
||`log_time`|Takes value `1` if time is to be plotted logarithmically and `0` if not.|None|
||`t(tau)`|Function for determining the timepoints (e.g. linearly or exponentially spaced). Must have that `t(0) = 0`.|s|
|`phys`|`L`|Length of the poroelastic domain.|m|
||`beta_E`|Weakening parameter.|Lg<sup>-1</sup>s<sup>-1</sup>|
||`E_min`|Minimum Young's modulus.|Pa|
||`D_m`|Diffusivity of solute.|m<sup>2</sup>s<sup>-1</sup>|
||`nu`|Poisson's ratio of material.|None|
||`mu`|Viscosity of fluid.|Pa s|
|`scales`|`t`|Timescale of weakening.|s|
||`k`|Scale for permeability (initial permeability).|m<sup>2</sup>|
||`c`|Scale for solute concentration (far-field).|Lg<sup>-1</sup>|
||`E`|Scale for Young's modulus (initial).|Pa|
||`v`|Scale for velocity. NOTE: this is only used if we prescribe a fixed flux and not a fixed pressure drop.|ms<sup>-1</sup>|
|`ics`|`phi_f`|Initial porosity.|None|
||`E`|Nondimensional initial Young's modulus.|None|
||`c`|Nondimensional initial solute concentration.|None|
||`u_s`|Nondimensional initial displacement.|None|
|`bcs`|`u_s_right`|Displacement on the right boundary.|None|
||`sigma_left`|Solid stress on the left boundary.|None|
||`Delta p`|Nondimensional pressure drop.|None|
||`p_f_right`|Pressure on the right boundary.|None|
||`c_minus`|(Optional) fixed solute concentration on the left boundary. Only if using Dirichlet conditions on `c`.|None|
||`c_plus`|(Optional) fixed solute concentration on the right boundary. Only if using Dirichlet conditions on `c`.|None|

We note that the variable `tau` varies on a linear scale and can be used to vary the timestep `delta_t`. For example, for an exponentially increasing time `t`, a function of the form `t(tau) = A * exp(tau - tau_star) + B` would be appropriate, for parameters `A`, `B` and `tau_star`. The function `t(tau)` must be written in C syntax (see [here](https://en.cppreference.com/c/header/math) for more details.).

## Running a simulation

The fastest way to run a simulation is to edit the file `scripts/sim/run_ndw_sim.py` and enter the `parent`, `trial` and `sub_trial` directory names on lines `17-19` of the file. For example, if your `params.json` file lies in `resources/phys/enzymatic/c_10_minus_2`, you would enter

```python
parent = "phys"
trial = "enzymatic"
sub_trial = "c_10_minus_2"
```

From here, remove the `ess_dict` parameter from line `47` so that it reads

```python
sim.solve()
```

Save the file and run from the terminal, ensuring you are in the `weakening` directory, using the following command:

```
python3 scripts/sim/run_ndw_sim.py
```

After the simulation has terminated, `data` and `plots` directories will be created inside your `resources` directory to store and visualise the simulation output.

However, for improved accuracy of the simulation, additional code can be run to generate an early-time similarity solution for the first timestep. To do this, firstly copy your `params.json` file into the same directory and rename the copy to `params_eta.json`. Here, $\\eta$ is the similarity variable, and is equal to $(1-x)/\\sqrt{t}$, where $x$ is the Eulerian spatial coordinate and $t$ is time. The file `params_eta.json` will be the input file for the early-time similarity solution code. Two additional parameters are needed for this simulation:

|Category|Parameter|Description|Unit|
|--|--|----------|--|
|`comp`|`N_eta`|Number of gridpoints in `eta` space.|None|
|`phys`|`L_eta`|The maximal value of `eta`.|None|

The value `L_eta` ($L_\\eta$) can be calculated as $L_{\\eta} = 1 / \\sqrt{t_1}$, where $t_1$ is the first timepoint of the simulation (after $t = 0$).

Once `params_eta.json` is created, edit the file `scripts/sim/run_ess_sim.py` and enter the `parent`, `trial` and `sub_trial` for your parameters. After saving this file, return to the `weakening` directory and run the following command from the terminal:

```
python3 scripts/sim/run_ess_sim.py
```

This generates an early-time similarity solution within your `resources` directory. Next, edit `scripts/sim/run_ndw_sim.py` and ensure that line `47` reads

```python
sim.solve(ess_dict)
```

Once this file is saved, return to the `weakening` directory from the terminal and run

```
python3 scripts/sim/run_ndw_sim.py
```

to generate more accurate simulation results.
