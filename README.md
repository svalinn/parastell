## ParaStell
Parametric 3D CAD modeling tool for stellarator fusion devices. This toolset takes VMEC plasma equilibrium data to build intra-magnet components of varying thickness from a user-specified radial build, as well as corresponding coil filament point-locus data to build magnet coils of user-specified cross-section.

## Dependencies
This tool depends on:

- [The VMEC tools](https://github.com/aaroncbader/pystell_uw) developed by @aaroncbader 
- [CadQuery](https://cadquery.readthedocs.io/en/latest/installation.html)
- [MOAB](https://bitbucket.org/fathomteam/moab/src/master/)
- [Coreform Cubit](https://coreform.com/products/downloads/) or [Cubit](https://cubit.sandia.gov/downloads/) by Sandia National Laboratories
- [cad-to-dagmc](https://pypi.org/project/cad-to-dagmc/)
- [Gmsh](https://pypi.org/project/gmsh/)
- [YAML](https://pyyaml.org/wiki/PyYAML)
## Install with Mamba

This guide will use the mamba package manager to install dependencies in a conda environment. Conda allows for easily installing and switching between different versions of software packages through the use of [environments](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).

If you have not already installed conda, you can use one of the following installers to do so: [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Anaconda](https://www.anaconda.com/), or [Miniforge](https://github.com/conda-forge/miniforge).

Mamba is available through conda with `conda install -c conda-forge mamba`. Begin by creating a new conda environment with mamba installed.

```
conda create -n parastell_env
conda activate parastell_env
conda install -c conda-forge mamba
```

The subsequent mamba and pip install commands should be run with this environment activated.

### Install Dependencies using Mamba and Pip
Install ParaStell dependencies available on conda-forge.

```
mamba install -c conda-forge cadquery moab
```

Install pystell_uw dependencies.

```
mamba install matplotlib
```

Pip install remaining pystell_uw dependency.

```
pip install netCDF4
```

Pip install remaining ParaStell dependencies.

```
pip install cad-to-dagmc
pip install gmsh
pip install pyyaml
```
### Coreform Cubit
Download and install version 2023.11 from [Coreform's Website](https://coreform.com/products/downloads/), then add the /Coreform-Cubit-2023.11/bin/ directory to the `PYTHONPATH` by adding a line to the .bashrc file like the following:

```
export PYTHONPATH=$PYTHONPATH:$HOME/Coreform-Cubit-2023.11/bin/
```

Replace $HOME with the path to the Cubit directory on your system. Additional information about adding modules to the `PYTHONPATH` can be found [here](https://www.tutorialspoint.com/How-to-set-python-environment-variable-PYTHONPATH-on-Linux).
While it is possible to use ParaStell with older versions of Cubit, additional steps not in this guide may be required.

If you do not have a Cubit license, you may be able to get one through [Cubit Learn](https://coreform.com/products/coreform-cubit/free-meshing-software/) at no cost.

### VMEC Tools
Download and extract the repository for pystell_uw using

```
git clone https://github.com/aaroncbader/pystell_uw.git
```

or download the and extract the zip from [pystell_uw](https://github.com/aaroncbader/pystell_uw). Once extracted, add the directory to the `PYTHONPATH`.

### Parastell
Download and extract the repository for parastell using

```
git clone git@github.com:svalinn/parastell.git
```

or download the zip from the repository home page. Once extracted, add the directory to the `PYTHONPATH`.


