from parastell.utils import *


def test_dagmc_renumbering():
    """Tests whether multiple DAGMC models are correctly combined, renumbering
    their IDs, by testing if:
        * the expected number of surfaces is created
        * the resulting largest surface ID is the expected value
        * the expected number of volumes is created
        * the resulting largest volume ID is the expected value
        * the correct material tags, in the correct order, are applied

    An example of three cubes is used for this test. two_cubes.h5m is two cubes
    which share a face. one_cube.h5m has one cube. They do not overlap.
    """
    num_surfaces_exp = 17
    max_surf_id_exp = 17
    num_vol_exp = 3
    max_vol_id_exp = 3

    mats_exp = ["iron", "tungsten", "air"]

    core_1 = core.Core()
    core_2 = core.Core()
    core_1.load_file("files_for_tests/two_cubes.h5m")
    core_2.load_file("files_for_tests/one_cube.h5m")

    combined_model = combine_dagmc_models([core_1, core_2])

    mats = [vol.material for vol in combined_model.volumes]

    assert len(combined_model.surfaces) == num_surfaces_exp
    assert max(combined_model.surfaces_by_id.keys()) == max_surf_id_exp
    assert len(combined_model.volumes) == num_vol_exp
    assert max(combined_model.volumes_by_id.keys()) == max_vol_id_exp
    assert all(mat in mats_exp for mat in mats)


def test_expand_list():
    """Tests utils.expand_list() to ensure returned arrays are the length
    expected, and contain the expected values, by testing if:
        * the expected entries are added to uniformly and non-uniformly spaced
          lists
        * entries are added when the requested size is less than or equal to
          that of the input list (no entries should be added)
    """
    # Make sure new entries are inserted as expected
    test_values = np.linspace(1, 10, 10)
    exp_expanded_list = np.linspace(1, 10, 19)
    expanded_list = expand_list(test_values, 19)
    assert np.allclose(exp_expanded_list, expanded_list)

    # Make sure no changes are made if list already has the requested number of
    # entries
    expanded_list = expand_list(test_values, 10)
    assert len(expanded_list) == len(test_values)
    assert np.allclose(expanded_list, test_values)

    # Make sure no changes are made if list has more than the requested number
    # of entries
    expanded_list = expand_list(test_values, 5)
    assert len(expanded_list) == len(test_values)
    assert np.allclose(expanded_list, test_values)

    # Make sure it works with unevenly spaced entries
    test_values = [1, 5, 6, 7, 10]
    exp_expanded_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    expanded_list = expand_list(test_values, 10)
    assert np.allclose(expanded_list, exp_expanded_list)

    # Make sure it works with unevenly spaced entries that are not
    # nicely divisible
    test_values = [1, 4.5, 6, 7, 10]
    expanded_list = expand_list(test_values, 5)
    assert len(expanded_list) == 5

    # int math makes this list one element longer than requested
    test_values = [1, 4.5, 6, 7, 10]
    expected_values = [1, 1.875, 2.75, 3.625, 4.5, 5.25, 6, 7, 8, 9, 10]
    expanded_list = expand_list(test_values, 10)
    assert len(expanded_list) == 11
    assert np.allclose(expected_values, expanded_list)


def test_stl_surfaces_to_cq_solid():
    """Tests utils.stl_surface_to_cq_solid() to verify that the correct number
    of faces are present in the CadQuery solid representation of the DAGMC
    volume, and that the volume of the DAGMC volume was preserved in the
    CadQuery solid by checking if:
      * the number of triangle handles belonging to the DAGMC volume is the
        same as the number of faces belonging to the CadQuery solid.
      * the volume of the DAGMC volume is close to the volume of the CadQuery
        solid (using math.isclose()).
    """
    vol_id = 1
    dagmc_model = pydagmc.Model("files_for_tests/one_cube.h5m")
    volume = dagmc_model.volumes_by_id[vol_id]
    dagmc_volume_volume = volume.volume

    with tempfile.NamedTemporaryFile(delete=True, suffix=".stl") as temp_file:
        stl_path = temp_file.name
        dagmc_model.mb.write_file(
            stl_path, output_sets=[s.handle for s in volume.surfaces]
        )
        cq_solid = stl_to_cq_solid(stl_path)

    cq_solid_volume = cq_solid.Volume()
    num_faces = len(cq_solid.Faces())
    num_tris = len(volume.triangle_handles)

    assert num_faces == num_tris
    assert np.isclose(dagmc_volume_volume, cq_solid_volume)


def test_ribs_from_kisslinger_format():
    """Tests that the example kisslinger format file is being read correctly by
    checking if:
      * The values for the toroidal angles match the original values, which
        are 64 evenly spaced values between 0 and 90.
      * The number of toroidal angles read from the file is correct.
      * The number of poloidal locations is correct.
      * The number of periods has been read correctly.
      * The shape of the custom rib data matches with the number of toroidal
        and poloidal angles expected.
      * The first and last ribs have the same R, Z data.
      * The first and second ribs do not have the same R, Z data.
      * Data is formatted correctly.
    """
    original_ribs_file = (
        Path("files_for_tests") / "kisslinger_file_example.txt"
    )
    scrambled_ribs_file = (
        Path("files_for_tests") / "kisslinger_file_scrambled.txt"
    )

    (
        _,
        _,
        _,
        _,
        original_surface_coords,
    ) = ribs_from_kisslinger_format(
        original_ribs_file,
        delimiter=" ",
        scale=1,
    )
    (
        toroidal_angles,
        num_toroidal_angles,
        num_poloidal_angles,
        periods,
        unscrambled_surface_coords,
    ) = ribs_from_kisslinger_format(
        scrambled_ribs_file,
        delimiter=" ",
        scale=1,
    )

    num_toroidal_angles_exp = 121
    num_poloidal_angles_exp = 121
    periods_exp = 4
    surface_coords_shape_exp = (121, 121, 2)

    assert np.allclose(np.linspace(0, 90, 121), toroidal_angles)
    assert num_toroidal_angles == num_toroidal_angles_exp
    assert num_poloidal_angles == num_poloidal_angles_exp
    assert periods == periods_exp
    assert unscrambled_surface_coords.shape == surface_coords_shape_exp
    assert np.allclose(
        unscrambled_surface_coords[0] - unscrambled_surface_coords[-1], 0
    )
    assert not np.allclose(
        unscrambled_surface_coords[0], unscrambled_surface_coords[1]
    )
    assert np.allclose(unscrambled_surface_coords, original_surface_coords)
