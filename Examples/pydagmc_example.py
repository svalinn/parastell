import numpy as np
import parastell.parastell as ps
from parastell.utils import merge_dagmc_files, enforce_helical_symmetry

ribs = 61
rib_pts = 63

toroidal_angles = np.linspace(0, 90, ribs)
poloidal_angles = np.linspace(0, 360, rib_pts)

challenge_thickness_matrix = enforce_helical_symmetry(
    np.random.rand(ribs, rib_pts) * 100
)
wall_s = 1.08

# Define plasma equilibrium VMEC file
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build
stellarator = ps.Stellarator(vmec_file)

ones = np.ones((ribs, rib_pts))

radial_build_dict = {
    "first_wall": {
        "thickness_matrix": ones * 0.2,
        "mat_tag": "iron",
    },
    "breeder": {
        "thickness_matrix": challenge_thickness_matrix,
        "mat_tag": "iron",
    },
    "back_wall": {
        "thickness_matrix": ones * 0.01,
        "mat_tag": "iron",
    },
    "shield": {
        "thickness_matrix": ones * 50,
        "mat_tag": "iron",
    },
    "vacuum_vessel": {
        "thickness_matrix": ones * 10,
        "mat_tag": "tungsten",
    },
    "test_1": {
        "thickness_matrix": ones * 0.1,
        "mat_tag": "tungsten",
    },
    "test_": {
        "thickness_matrix": ones * 5,
        "mat_tag": "tungsten",
    },
}

stellarator.construct_invessel_build(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    radial_build_dict,
    use_pydagmc=True,
    num_ribs=ribs * 3,
    num_rib_pts=rib_pts * 3,
)

stellarator.invessel_build.dag_model.write_file("dagmc.h5m")

stellarator.construct_magnets(
    "../tests/files_for_tests/coils.example", 10, 10, 90
)
stellarator.export_magnets()

stellarator.build_cubit_model(skip_imprint=False, legacy_faceting=False)
stellarator.export_dagmc("magnets.h5m")
merge_dagmc_files(["dagmc.h5m", "magnets.h5m"], "merged_dagmc.h5m")
