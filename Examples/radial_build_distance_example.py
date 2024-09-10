from parastell.radial_distance_utils import *

coils_file = "coils.example"
width = 40.0
thickness = 50.0
toroidal_extent = 90.0
vmec_file = "wout_vmec.nc"
vmec = read_vmec.VMECData(vmec_file)

magnet_set = magnet_coils.MagnetSet(
    coils_file, width, thickness, toroidal_extent
)

reordered_filaments = get_reordered_filaments(magnet_set)

build_magnet_surface(reordered_filaments)

toroidal_angles = np.linspace(0, 90, num=4)
poloidal_angles = np.linspace(0, 360, num=4)
wall_s = 1.08

radial_build_dict = {}

radial_build = ivb.RadialBuild(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    radial_build_dict,
    split_chamber=False,
)
build = ivb.InVesselBuild(vmec, radial_build, num_ribs=72, num_rib_pts=96)
build.populate_surfaces()
build.calculate_loci()
ribs = build.Surfaces["chamber"].Ribs
radial_distances = measure_radial_distance(ribs)
np.savetxt("radial_distances.csv", radial_distances, delimiter=",")
