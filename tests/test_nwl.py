from pathlib import Path
import os

import pytest
import openmc
from parastell import invessel_build
from parastell import nwl_utils
from parastell.utils import ribs_from_kisslinger_format
import numpy as np


files_to_remove = [
    "statepoint.1.h5",
    "summary.h5",
    "model.xml",
    "surface_source.h5",
    "nwl.png",
]

vmec_file = Path("files_for_tests") / "wout_vmec.nc"

ribs_file = Path("files_for_tests") / "kisslinger_file_example.txt"
(
    custom_surface_toroidal_angles,
    num_toroidal_angles,
    num_poloidal_angles,
    periods,
    custom_surface_rz_ribs,
) = ribs_from_kisslinger_format(ribs_file, delimiter=" ", scale=1.0)
poloidal_angles = np.linspace(0, 360, num_poloidal_angles)
rib_based_surface = invessel_build.RibBasedSurface(
    custom_surface_rz_ribs, custom_surface_toroidal_angles, poloidal_angles
)


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


def allclose_periodic(a, b, period=360.0):
    difference = np.abs(a - b)
    difference = np.minimum(difference, period - difference)

    return np.allclose(difference, np.zeros(difference.shape), atol=1e-3)


@pytest.fixture
def parastell_model(ref_surf):
    dagmc_filename = Path("files_for_tests") / "nwl_geom.h5m"
    source_mesh_filename = Path("files_for_tests") / "source_mesh.h5m"
    toroidal_extent = 15.0
    wall_s = 1.08
    strengths = np.load(Path("files_for_tests") / "strengths.npy")

    parastell_build_data = {
        "ref_surf": ref_surf,
        "dagmc_filename": dagmc_filename,
        "source_mesh_filename": source_mesh_filename,
        "toroidal_extent": toroidal_extent,
        "wall_s": wall_s,
        "strengths": strengths,
    }

    return parastell_build_data


@pytest.mark.parametrize("ref_surf", [vmec_file, rib_based_surface])
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

    ref_surf = parastell_model["ref_surf"]
    dagmc_filename = parastell_model["dagmc_filename"]
    source_mesh_filename = parastell_model["source_mesh_filename"]
    toroidal_extent = parastell_model["toroidal_extent"]
    wall_s = parastell_model["wall_s"]
    strengths = parastell_model["strengths"]

    num_parts = 1_000
    neutron_energy = 14.1e6 * 1.60218e-19 * 1e-6  # eV to MJ
    neutron_power = neutron_energy * np.sum(strengths)

    # CI needs a functional cross-section library for OpenMC to run
    openmc.Materials.cross_sections = (
        Path("files_for_tests") / "cross_sections" / "cross_sections.xml"
    )

    source_file = nwl_utils.fire_rays(
        dagmc_filename,
        source_mesh_filename,
        toroidal_extent,
        strengths,
        num_parts,
    )

    assert Path(source_file_exp).exists()

    nwl_mat, toroidal_bins, poloidal_bins, area_mat = nwl_utils.compute_nwl(
        source_file,
        ref_surf,
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

    nwl_utils.plot_nwl(
        nwl_mat, toroidal_bins, poloidal_bins, filename=plot_filename_exp
    )

    assert Path(plot_filename_exp).exists()

    remove_files()


@pytest.mark.parametrize("ref_surf", [vmec_file, rib_based_surface])
def test_flux_coordinate_calculation(parastell_model):
    """Tests whether the flux coordinate computation routine is correct, by
    testing if:
        * Calculated flux coordinates match known correct answers
    """
    toroidal_angles_exp = np.repeat([0.0, 22.5, 45.0, 67.5, 90.0], 4)
    poloidal_angles_exp = np.tile([0.0, 90.0, 180.0, 270.0], 5)

    remove_files()

    ref_surf = parastell_model["ref_surf"]
    wall_s = parastell_model["wall_s"]

    coords = np.array(
        [
            [1425.928700169618, 0.0, 0.0],
            [1188.3252024887156, 0.0, 358.5064203208586],
            [1284.4131097301138, 0.0, 4.561644170634904e-14],
            [1188.3252024887156, 0.0, -358.5064203208586],
            [1205.03265616365, 499.1408692854585, -108.15593369493526],
            [1204.5136312392501, 498.9258821225623, 47.80647441319307],
            [799.5184825738327, 331.17139885003854, -124.09820124422208],
            [835.9278821816554, 346.2526659654604, -289.45466073509107],
            [875.5124086972241, 875.5124086972241, -1.231480251650162e-14],
            [676.2888112463809, 676.2888112463809, 113.96140110315854],
            [496.74462745068155, 496.74462745068143, 1.349053166093886e-14],
            [676.2888112463809, 676.2888112463809, -113.9614011031586],
            [499.1408692854586, 1205.03265616365, 108.15593369493526],
            [346.25266596546044, 835.9278821816554, 289.45466073509095],
            [331.1713988500386, 799.5184825738327, 124.09820124422207],
            [498.92588212256237, 1204.5136312392501, -47.806474413193015],
            [8.731295092375343e-14, 1425.928700169618, 3.576922228891618e-14],
            [7.276393277869679e-14, 1188.3252024887156, 358.50642032085864],
            [7.864762018069409e-14, 1284.413109730114, 1.244299282564651e-13],
            [7.276393277869679e-14, 1188.3252024887156, -358.50642032085864],
        ]
    )
    num_threads = os.cpu_count()
    conv_tol = 1e-6

    toroidal_angles, poloidal_angles = nwl_utils.compute_flux_coordinates(
        ref_surf, wall_s, coords, num_threads, conv_tol
    )

    assert allclose_periodic(np.rad2deg(toroidal_angles), toroidal_angles_exp)
    assert allclose_periodic(np.rad2deg(poloidal_angles), poloidal_angles_exp)

    remove_files()
