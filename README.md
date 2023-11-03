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

## Install with Cubit 2023.11

### Parastell
Download and extract repository for parastell using

```
git clone git@github.com:svalinn/parastell.git
```

or download the zip from the repository home page. Once extracted, add the directory to the `PYTHONPATH` by adding a line to the .bashrc file like the following:

```
export PYTHONPATH=$PYTHONPATH:$HOME/parastell/
```

Replace $HOME with the path to the parastell directory on your system. Additional information about adding modules to the `PYTHONPATH` can be found [here](https://www.tutorialspoint.com/How-to-set-python-environment-variable-PYTHONPATH-on-Linux).

### VMEC Tools
Download and extract the repository for pystell_uw using

```
git clone https://github.com/aaroncbader/pystell_uw.git
```

or download the and extract the zip from [pystell_uw](https://github.com/aaroncbader/pystell_uw). Once extracted, add the directory to the `PYTHONPATH`.

### Coreform Cubit
Download and install version 2023.11 from [Coreform's Website](https://coreform.com/products/downloads/), then add the /Coreform-Cubit-2023.11/bin/ directory to the `PYTHONPATH`.

If you do not have a Cubit license, you may be able to get one through [Cubit Learn](https://coreform.com/products/coreform-cubit/free-meshing-software/) at no cost.

### Create a Conda Environment for ParaStell
This guide uses the mamba package manager. If the mamba package manager is not available, using conda in place of mamba in the following commands should also work.

```
mamba create -n parastell_env
mamba activate parastell_env
``````

### Install Dependencies using Mamba and Pip
Install ParaStell dependencies available on conda-forge.

```
mamba install -c conda-forge cadquery moab
```

Install pystell_uw dependencies available on conda-forge.

```
mamba install -c conda-forge matplotlib
```

Pip install remaining pystell_uw dependency.

```
pip install netCDF4
```

Pip install remaining ParaStell dependencies.

```
pip install cad-to-dagmc
pip install gmsh
```
