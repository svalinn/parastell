from pathlib import Path

import pytest
from parastell import parastell
from parastell.nwl_utils import *
from parastell.cubit_utils import tag_surface
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


files_to_remove = [
    "statepoint.1.h5",
    "summary.h5",
    "model.xml",
    "surface_source.h5",
    "nwl.png",
]


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


@pytest.fixture
def parastell_model():
    vmec_file = Path("files_for_tests") / "wout_vmec.nc"
    dagmc_filename = Path("files_for_tests") / "nwl_geom.h5m"
    source_mesh_filename = Path("files_for_tests") / "source_mesh.h5m"
    toroidal_extent = 15.0
    wall_s = 1.08
    strengths = np.load(Path("files_for_tests") / "strengths.npy")

    parastell_build_data = {
        "vmec_file": vmec_file,
        "dagmc_filename": dagmc_filename,
        "source_mesh_filename": source_mesh_filename,
        "toroidal_extent": toroidal_extent,
        "wall_s": wall_s,
        "strengths": strengths,
    }

    return parastell_build_data


def test_nwl_io(parastell_model):
    """Tests whether the I/O of the NWL workflow behaves as expected, by
    testing if:
        * the expected surface source file is produced
        * the computed arrays are of the correct shape
        * the expected plot file is produced
    """
    source_file_exp = "surface_source.h5"
    num_bins_exp = 61
    plot_filename_exp = "nwl.png"

    remove_files()

    vmec_file = parastell_model["vmec_file"]
    dagmc_filename = parastell_model["dagmc_filename"]
    source_mesh_filename = parastell_model["source_mesh_filename"]
    toroidal_extent = parastell_model["toroidal_extent"]
    wall_s = parastell_model["wall_s"]
    strengths = parastell_model["strengths"]

    num_parts = 1_000
    neutron_energy = 14.1e6 * 1.60218e-19 * 1e-6  # eV to MJ
    neutron_power = neutron_energy * np.sum(strengths)

    source_file = fire_rays(
        dagmc_filename,
        source_mesh_filename,
        toroidal_extent,
        strengths,
        num_parts,
    )

    assert Path(source_file_exp).exists()

    nwl_mat, toroidal_bins, poloidal_bins, area_mat = compute_nwl(
        source_file,
        vmec_file,
        wall_s,
        toroidal_extent,
        neutron_power,
        num_toroidal_bins=num_bins_exp,
        num_poloidal_bins=num_bins_exp,
    )

    assert nwl_mat.shape == (num_bins_exp, num_bins_exp)
    assert len(toroidal_bins) == num_bins_exp
    assert len(poloidal_bins) == num_bins_exp
    assert area_mat.shape == (num_bins_exp, num_bins_exp)

    plot_nwl(nwl_mat, toroidal_bins, poloidal_bins, filename=plot_filename_exp)

    assert Path(plot_filename_exp).exists()

    remove_files()
