from pathlib import Path

import numpy as np

# import this before read_vmec to deal with conflicting 
# dependencies correctly
import src.invessel_build as ivb

import read_vmec


if Path('plasma.step').exists():
    Path.unlink('plasma.step')
if Path('sol.step').exists():
    Path.unlink('sol.step')
if Path('component.step').exists():
    Path.unlink('component.step')
if Path('dagmc.h5m').exists():
    Path.unlink('dagmc.h5m')

vmec_file = Path('files_for_tests') / 'wout_vmec.nc'
vmec = read_vmec.vmec_data(vmec_file)

toroidal_angles_exp = [0.0, 30.0, 60.0, 90.0]
poloidal_angles_exp = [0.0, 120.0, 240.0, 360.0]
radial_build = {
    'component': {
        'thickness_matrix': np.ones(
            (len(toroidal_angles_exp), len(poloidal_angles_exp))
        )*10
    }
}
wall_s_exp = 1.08
repeat_exp = 0
num_ribs_exp = 61
num_rib_pts_exp = 61
scale_exp = 100
plasma_mat_tag_exp = 'Vacuum'
sol_mat_tag_exp = 'Vacuum'

invessel_build = ivb.InVesselBuild(
    vmec, toroidal_angles_exp, poloidal_angles_exp, radial_build, wall_s_exp,
    repeat=repeat_exp, num_ribs=num_ribs_exp, num_rib_pts=num_rib_pts_exp,
    scale=scale_exp, plasma_mat_tag=plasma_mat_tag_exp,
    sol_mat_tag=sol_mat_tag_exp
)


def test_ivb_basics():

    invessel_build.populate_surfaces()

    assert np.allclose(invessel_build.toroidal_angles, toroidal_angles_exp)
    assert np.allclose(invessel_build.poloidal_angles, poloidal_angles_exp)
    assert invessel_build.wall_s == wall_s_exp
    assert len(invessel_build.radial_build.keys()) == 3
    assert (
        invessel_build.radial_build['plasma']['mat_tag'] == 'Vacuum'
    )
    assert (
        invessel_build.radial_build['sol']['mat_tag'] == 'Vacuum'
    )
    assert invessel_build.repeat == repeat_exp
    assert invessel_build.num_ribs == num_ribs_exp
    assert invessel_build.num_rib_pts == num_rib_pts_exp
    assert invessel_build.scale == scale_exp


def test_ivb_construction():

    invessel_build.calculate_loci()
    invessel_build.generate_components()

    assert len(invessel_build.Components) == 3


def test_ivb_exports():
    
    invessel_build.export_step()
    """
    CAD-to-DAGMC export does not currently work; might be an incompatibility
    between CadQuery and CAD-to-DAGMC versions.

    invessel_build.export_cad_to_dagmc()
    """
    assert Path('plasma.step').exists() 
    assert Path('sol.step').exists() 
    assert Path('component.step').exists() 

    Path.unlink('plasma.step')
    Path.unlink('sol.step')
    Path.unlink('component.step')
    Path.unlink('stellarator.log')
