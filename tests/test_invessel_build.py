from pathlib import Path

import numpy as np
import pytest

# import this before read_vmec to deal with conflicting 
# dependencies correctly
import parastell.invessel_build as ivb

import src.pystell.read_vmec as read_vmec


def remove_files():
    
    if Path('plasma.step').exists():
        Path.unlink('plasma.step')
    if Path('sol.step').exists():
        Path.unlink('sol.step')
    if Path('component.step').exists():
        Path.unlink('component.step')
    if Path('dagmc.h5m').exists():
        Path.unlink('dagmc.h5m')


@pytest.fixture
def radial_build():

    toroidal_angles = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles = [0.0, 120.0, 240.0, 360.0]
    wall_s = 1.08
    radial_build_dict = {
        'component': {
            'thickness_matrix': np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )*10
        }
    }

    radial_build_obj = ivb.RadialBuild(
        toroidal_angles, poloidal_angles, wall_s, radial_build_dict
    )

    return radial_build_obj


@pytest.fixture
def invessel_build(radial_build):
    
    vmec_file = Path('files_for_tests') / 'wout_vmec.nc'
    vmec = read_vmec.vmec_data(vmec_file)

    ivb_obj = ivb.InVesselBuild(vmec, radial_build)

    return ivb_obj


def test_ivb_basics(invessel_build):

    toroidal_angles_exp = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles_exp = [0.0, 120.0, 240.0, 360.0]
    num_components_exp = 3
    wall_s_exp = 1.08
    repeat_exp = 0
    num_ribs_exp = 11
    num_rib_pts_exp = 67
    scale_exp = 100
    plasma_mat_tag_exp = 'Vacuum'
    sol_mat_tag_exp = 'Vacuum'

    remove_files()
    
    invessel_build.populate_surfaces()

    assert np.allclose(
        invessel_build.radial_build.toroidal_angles, toroidal_angles_exp
    )
    assert np.allclose(
        invessel_build.radial_build.poloidal_angles, poloidal_angles_exp
    )
    assert (
        len(invessel_build.radial_build.radial_build.keys()) ==
        num_components_exp
    )
    assert invessel_build.radial_build.wall_s == wall_s_exp
    assert (
        invessel_build.radial_build.radial_build['plasma']['mat_tag'] ==
        plasma_mat_tag_exp
    )
    assert (
        invessel_build.radial_build.radial_build['sol']['mat_tag'] ==
        sol_mat_tag_exp
    )
    assert invessel_build.repeat == repeat_exp
    assert invessel_build.num_ribs == num_ribs_exp
    assert invessel_build.num_rib_pts == num_rib_pts_exp
    assert invessel_build.scale == scale_exp

    remove_files()


def test_ivb_construction(invessel_build):

    num_components_exp = 3

    remove_files()

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()

    assert len(invessel_build.Components) == num_components_exp

    remove_files()


def test_ivb_exports(invessel_build):
    
    remove_files()

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()
    invessel_build.export_step()
    invessel_build.export_cad_to_dagmc()

    assert Path('plasma.step').exists() 
    assert Path('sol.step').exists() 
    assert Path('component.step').exists()
    assert Path('dagmc.h5m').exists()

    remove_files() 
