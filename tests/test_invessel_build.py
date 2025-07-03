from pathlib import Path

import numpy as np
import pytest

# import this before read_vmec to deal with conflicting
# dependencies correctly
import parastell.invessel_build as ivb
from parastell.cubit_utils import (
    check_cubit_installation,
    create_new_cubit_instance,
)
from parastell.utils import ribs_from_kisslinger_format

import pystell.read_vmec as read_vmec


files_to_remove = [
    "chamber.step",
    "component_1.step",
    "component_2.step",
    "component_3.step",
    "ivb_mesh.h5m",
    "step_import.log",
    "stellarator.log",
]


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


vmec_file = Path("files_for_tests") / "wout_vmec.nc"
vmec_surface = ivb.VMECSurface(read_vmec.VMECData(vmec_file))

ribs_file = Path("files_for_tests") / "kisslinger_file_example.txt"
(
    custom_surface_toroidal_angles,
    num_toroidal_angles,
    num_poloidal_angles,
    periods,
    custom_surface_rz_ribs,
) = ribs_from_kisslinger_format(ribs_file, delimiter=" ", scale=1.0)
poloidal_angles = np.linspace(0, 360, num_poloidal_angles)
rib_based_surface = ivb.RibBasedSurface(
    custom_surface_rz_ribs, custom_surface_toroidal_angles, poloidal_angles
)


@pytest.fixture
def radial_build():
    toroidal_angles = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles = [0.0, 120.0, 240.0, 360.0]
    wall_s = 1.08
    radial_build_dict = {
        "component_1": {
            "thickness_matrix": np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )
            * 10
        },
        "component_2": {
            "thickness_matrix": np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )
            * 10
        },
        "component_3": {
            "thickness_matrix": np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )
            * 10
        },
    }

    radial_build_obj = ivb.RadialBuild(
        toroidal_angles, poloidal_angles, wall_s, radial_build_dict
    )

    return radial_build_obj


@pytest.fixture
def invessel_build(radial_build, ref_surf):
    num_ribs = 11

    ivb_obj = ivb.InVesselBuild(ref_surf, radial_build, num_ribs=num_ribs)

    return ivb_obj


@pytest.mark.parametrize("ref_surf", [vmec_surface, rib_based_surface])
def test_ivb_basics(invessel_build):
    """Tests whether InVesselBuild arguments are instantiated as expected, by
    testing if:
        * after being set, member variables match inputs
    """
    toroidal_angles_exp = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles_exp = [0.0, 120.0, 240.0, 360.0]
    num_components_exp = 4
    wall_s_exp = 1.08
    repeat_exp = 0
    num_ribs_exp = 11
    num_rib_pts_exp = 61
    scale_exp = 100
    chamber_mat_tag_exp = "Vacuum"

    remove_files()

    assert np.allclose(
        invessel_build.radial_build.toroidal_angles, toroidal_angles_exp
    )
    assert np.allclose(
        invessel_build.radial_build.poloidal_angles, poloidal_angles_exp
    )
    assert (
        len(invessel_build.radial_build.radial_build.keys())
        == num_components_exp
    )
    assert invessel_build.radial_build.wall_s == wall_s_exp
    assert (
        invessel_build.radial_build.radial_build["chamber"]["mat_tag"]
        == chamber_mat_tag_exp
    )
    assert invessel_build.repeat == repeat_exp
    assert invessel_build.num_ribs == num_ribs_exp
    assert invessel_build.num_rib_pts == num_rib_pts_exp
    assert invessel_build.scale == scale_exp

    remove_files()


@pytest.mark.parametrize("ref_surf", [vmec_surface, rib_based_surface])
def test_ivb_cadquery_construction(invessel_build):
    """Tests whether the InVesselBuild CadQuery workflow functions as
    expected, by testing if:
        * the correct number of components were assembled
        * rib coordinates have the correct dimension
        * rib coordinates are defined by floating point numbers
    """
    num_components_exp = 4
    len_loci_pt_exp = 3

    remove_files()

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()

    rib_loci = invessel_build.Surfaces["chamber"].get_loci()[0]

    assert len(invessel_build.Components) == num_components_exp
    assert len(rib_loci[0]) == len_loci_pt_exp
    assert isinstance(rib_loci[0][0], float)

    remove_files()


@pytest.mark.parametrize("ref_surf", [vmec_surface, rib_based_surface])
def test_ivb_pydagmc_construction(invessel_build):
    """Tests whether the InVesselBuild PyDAGMC workflow functions as
    expected, by testing if:
        * the correct number of volumes and surfaces are produced
    """
    num_volumes_exp = 3
    num_surfaces_exp = 10

    invessel_build.use_pydagmc = True
    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()

    assert num_surfaces_exp == len(invessel_build.dag_model.surfaces)
    assert num_volumes_exp == len(invessel_build.dag_model.volumes)


@pytest.mark.parametrize("ref_surf", [vmec_surface, rib_based_surface])
def test_ivb_exports(invessel_build):
    """Tests whether the InVesselBuild CadQuery workflow's export
    functionality behaves as expected, by testing if:
        * the expected STEP are produced
        * if Cubit is correctly installed, the expected H5M file is produced
        * the expected H5M files are produced from the MOAB and Gmsh meshing
          routines
    For the ReferenceSurface types paramaterized above.

    The Cubit-enabled portion of this test is skipped if Cubit cannot be
    imported.
    """
    remove_files()

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()
    invessel_build.export_step()

    assert Path("chamber.step").exists()
    assert Path("component_1.step").exists()
    assert Path("component_2.step").exists()
    assert Path("component_3.step").exists()

    if check_cubit_installation():
        create_new_cubit_instance()

        invessel_build.mesh_components_cubit(
            ["chamber", "component_1", "component_3"]
        )
        invessel_build.export_mesh_cubit("ivb_mesh")
        assert Path("ivb_mesh.h5m").exists()

    remove_files()

    num_surfaces_exp = 4
    gap_map_exp = [False, True, False]

    invessel_build.mesh_components_moab(["component_1", "component_3"])
    invessel_build.export_mesh_moab("ivb_mesh")
    assert Path("ivb_mesh.h5m").exists()
    assert len(invessel_build.moab_mesh.surfaces) == num_surfaces_exp
    assert invessel_build.moab_mesh.gap_map == gap_map_exp

    remove_files()

    invessel_build.mesh_components_gmsh(
        ["chamber", "component_1", "component_3"]
    )
    invessel_build.export_mesh_gmsh("ivb_mesh")
    assert Path("ivb_mesh.h5m").exists()

    remove_files()
