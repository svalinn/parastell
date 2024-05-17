## Manually Installing Python Dependencies with Mamba

This guide will use the mamba package manager to install Python dependencies. Mamba is a faster, more reliable conda alternative that is fully compatible with conda packages. Mamba is available via conda (note that Miniforge ships with mamba already installed).

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