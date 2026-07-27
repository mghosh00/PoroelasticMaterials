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
||`E_mino`|||
