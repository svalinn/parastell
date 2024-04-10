# ParaStell
Parametric 3-D CAD modeling toolset for stellarator fusion devices. This open-source Python package uses plasma equilibrium VMEC data and CadQuery to model in-vessel components of varying thickness in low-fidelity from a user-specified radial build. Furthermore, coil filament point-locus data and Coreform Cubit are used to model magnet coils of user-specified cross-section. Additional neutronics support includes the use of VMEC data and MOAB to generate tetrahedral neutron source definitions and Coreform Cubit to generate DAGMC geometries for use in Monte Carlo radiation transport software.

## Dependencies
ParaStell depends on:

- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [NumPy](https://numpy.org/install/)
- [SciPy](https://scipy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [Coreform Cubit](https://coreform.com/products/downloads/), version 2023.11
- [CadQuery](https://cadquery.readthedocs.io/en/latest/installation.html)
- [MOAB](https://bitbucket.org/fathomteam/moab/src/master/)
- [PyStell-UW](https://github.com/aaroncbader/pystell_uw) developed by @aaroncbader 
- [CAD-to-DAGMC](https://github.com/fusion-energy/cad_to_dagmc)

### Install Python Dependencies

This guide will use the mamba package manager to install Python dependencies in a conda environment. Conda allows for easily installing and switching between different versions of software packages through the use of [environments](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).

If you have not already installed conda, you can use one of the following installers to do so:
- [Miniforge](https://github.com/conda-forge/miniforge)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://www.anaconda.com/)

Mamba is a faster, more reliable conda alternative that is fully compatible with conda packages. Mamba is available via conda (note that Miniforge ships with mamba already installed).

Begin by creating a new conda environment, installing mamba if needed. Note that ParaStell's dependencies are sensitive to Python version; ensure that Python 3.11.6 is installed.

```
conda create --name parastell_env python=3.11.6
conda activate parastell_env
conda install -c conda-forge mamba
```

The subsequent mamba and pip install commands should be run with this environment activated.

Mamba install ParaStell and PyStell-UW Python dependencies available on `conda-forge`:

```
mamba install -c conda-forge numpy scipy scikit-learn cadquery cad_to_dagmc matplotlib
mamba install -c conda-forge moab=5.5.0=nompi_tempest_*
```

Pip install the remaining ParaStell and PyStell-UW Python dependencies:

```
pip install netCDF4
pip install pyyaml
```

### Install Coreform Cubit
Download and install version 2023.11 from [Coreform's Website](https://coreform.com/products/downloads/), then add the `/Coreform-Cubit-2023.11/bin/` directory to your `PYTHONPATH` by adding a line similar to the following to your `.bashrc` file:

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/Coreform-Cubit-2023.11/bin/
```

Replace `$HOME` with the path to the Coreform Cubit directory on your system. Additional information about adding modules to your `PYTHONPATH` can be found [here](https://www.tutorialspoint.com/How-to-set-python-environment-variable-PYTHONPATH-on-Linux).
While it is possible to use ParaStell with older versions of Cubit, additional steps not in this guide may be required.

If you do not have a Coreform Cubit license, you may be able to get one through [Cubit Learn](https://coreform.com/products/coreform-cubit/free-meshing-software/) at no cost.

### Install PyStell-UW
Download and extract the PyStell-UW repository:

```bash
git clone https://github.com/aaroncbader/pystell_uw.git
```

or download and extract the ZIP file from [PyStell-UW](https://github.com/aaroncbader/pystell_uw). Once extracted, add the repository directory to your `PYTHONPATH`.

## Install ParaStell
Download and extract the ParaStell repository:

```bash
git clone git@github.com:svalinn/parastell.git
```

or download the ZIP file from the repository home page. Once extracted, add the repository directory to your `PYTHONPATH`.
