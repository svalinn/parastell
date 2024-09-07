from parastell.radial_distance_utils import *

coils_file = "coils_wistelld.txt"
circ_cross_section = ["circle", 25]
toroidal_extent = 90.0
vmec_file = "plasma_wistelld.nc"
vmec = read_vmec.VMECData(vmec_file)

magnet_set = magnet_coils.MagnetSet(
    coils_file, circ_cross_section, toroidal_extent
)

reordered_filaments = get_reordered_filaments(magnet_set)

build_magnet_surface(reordered_filaments)

toroidal_angles = np.linspace(0, 90, 72)
poloidal_angles = np.linspace(0, 360, 96)
wall_s = 1.08

radial_build_dict = {}

radial_build = ivb.RadialBuild(
    toroidal_angles, poloidal_angles, wall_s, radial_build_dict
)
build = ivb.InVesselBuild(vmec, radial_build, split_chamber=False)
build.populate_surfaces()
build.calculate_loci()
ribs = build.Surfaces["chamber"].Ribs
radial_distances = measure_radial_distance(ribs)
np.savetxt("radial_distances.csv", radial_distances, delimiter=",")
