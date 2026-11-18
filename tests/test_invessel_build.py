from pathlib import Path

import numpy as np
import cadquery as cq
import pytest

import parastell.invessel_build as ivb
from parastell.cubit_utils import (
    check_cubit_installation,
    create_new_cubit_instance,
)
from parastell.utils import ribs_from_kisslinger_format, normalize
from parastell.pystell import read_vmec


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
) = ribs_from_kisslinger_format(
    ribs_file, delimiter=" ", scale=1.0, format=False
)
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


##### Port Feature Testing #####
class TorusSurface(ivb.ReferenceSurface):
    """Analytic torus: major radius R0, minor radius a. Ignores s."""

    def __init__(self, R0, a):
        super().__init__()
        self.R0, self.a = R0, a

    def angles_to_xyz(self, toroidal_angle, poloidal_angles, s, scale):
        thetas = np.asarray(poloidal_angles, dtype=float)
        R = self.R0 + self.a * np.cos(thetas)
        return np.column_stack([
            R * np.cos(toroidal_angle),
            R * np.sin(toroidal_angle),
            self.a * np.sin(thetas)
        ]) * scale

R0, A, RADIUS = 500.0, 100.0, 20.0
PERIOD = 45.0
THICK = {"first_wall" : 5.0, "blanket" : 30.0, "shield" : 20.0}
TOTAL_THICK = sum(THICK.values())

# Geometry helpers (angles in degrees)
def surf_normal(theta, phi):
    th, ph = np.deg2rad(theta), np.deg2rad(phi)
    return np.array([
        np.cos(th) * np.cos(ph),
        np.cos(th) * np.sin(ph),
        np.sin(th)
    ])

def poloidal_tangent(theta, phi):
    th, ph = np.deg2rad(theta), np.deg2rad(phi)
    return np.array([
        -np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), np.cos(th)
    ])

def toroidal_tangent(phi):
    ph = np.deg2rad(phi)
    return np.array([-np.sin(ph), np.cos(ph), 0.0])

def tilt(n, t, alpha_deg):
    al = np.deg2rad(alpha_deg)
    return np.cos(al) * n + np.sin(al) * t

def rot_z(deg):
    d = np.deg2rad(deg)
    return np.array([
        [np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]
    ])

def torus_xyz(theta, phi):
    th, ph = np.deg2rad(theta), np.deg2rad(phi)
    R = R0 + A * np.cos(th)
    return np.array([R * np.cos(ph), R * np.sin(ph), A * np.sin(th)])

def brute_force_behind(phi, theta, axis, r, num=801):
    """Independent oracle: dense sampling of the torus near the base point,
    exact footprint (lateral <= r). Returns how far behind the base plaine the
    surface reaches inside the footprint.
    """

    axis = normalize(axis)
    ph0, th0 = np.deg2rad(phi), np.deg2rad(theta)
    base = torus_xyz(theta, phi)
    W = 4.0 * r
    TH, PH = np.meshgrid(
        th0 + np.linspace(-W / A, W / A, num),
        ph0 + np.linspace(-W / (R0 - A), W / (R0 - A), num)
    )
    Rr = R0 + A * np.cos(TH)
    pts = np.stack(
        [Rr * np.cos(PH), Rr * np.sin(PH), A * np.sin(TH)], axis=-1
    ).reshape(-1, 3)
    rel = pts - base
    axial = rel @ axis
    lateral = np.linalg.norm(rel - np.outer(axial, axis), axis=1)
    return max(0.0, -axial[lateral <= r].min())

# Constructors
def make_port(tor, polar, azimuth, pol=0.0, radius=10.0, **kw):
    return ivb.Port(
        TorusSurface(R0, A),
        1.0,
        tor, pol,
        orientation_polar_angle=polar,
        orientation_azimuthal_angle=azimuth,
        radius=radius,
        scale=1.0,
        **kw
    )

def port_from_axis(axis, phi, theta, radius=RADIUS, **kw):
    """Build a `Port` whose world-space axis is `axis`, by converting it to
    polar/azimuthal angles in the `Port`'s local frame.
    """

    a = normalize(np.asarray(axis, dtype=float))
    ph = np.deg2rad(phi)
    x = a @ np.array([np.cos(ph), np.sin(ph), 0.0])
    y = a @ np.array([-np.sin(ph), np.cos(ph), 0.0])
    polar = np.degrees(np.arccos(np.clip(a[2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(y, x))
    return ivb.Port(
        TorusSurface(R0, A),
        1.0,
        phi, theta,
        orientation_polar_angle=polar,
        orientation_azimuthal_angle=azimuth,
        radius=radius,
        scale=1.0,
        **kw
    )

def build(ports, repeat=0, num_ribs=16, num_rib_pts=41, resolve=True):
    """`InVesselBuild` on a circular torus. Layer thicknesses are constant,
    and on a circular cross-section the in-plane rib normal equals the surface
    normal, so layer boundaries are exact concentroic tori. Rib resolution
    defaults to a coarse grid for speed.
    """

    tor = [0.0, 15.0, 30.0, 45.0]
    pol = [0.0, 90.0, 180.0, 270.0, 360.0]
    layers = {
        name: {"thickness_matrix" : np.full((len(tor), len(pol)), t)}
        for name, t in THICK.items()
    }
    rb = ivb.RadialBuild(tor, pol, 1.0, layers)
    ivbuild = ivb.InVesselBuild(
        TorusSurface(R0, A),
        rb,
        scale=1.0,
        repeat=repeat,
        num_ribs=num_ribs,
        num_rib_pts=num_rib_pts,
        ports=ports
    )
    ivbuild.populate_surfaces()
    ivbuild.calculate_loci()

    if resolve:
        ivbuild.resolve_ports(ivbuild.radial_build.toroidal_angles[-1])

    return ivbuild

def _unresolved_build(ports):
    return build(ports, resolve=False)

def _box(center, side=4.0):
    h = side / 2.0
    return cq.Solid.makeBox(
        side, side, side, cq.Vector(*(np.asarray(center, dtype=float) - h))
    )

def _clearance_build(repeat=0):
    return build(
        [dict(
            base_toroidal_angle=22.5,
            base_poloidal_angle=30.0,
            radius=8.0,
            length=80.0
        )],
        repeat=repeat
    )

def _on_axis(port, depth=20.0):
    return port.base_point + depth * port.orientation


@pytest.fixture(scope="module")
def clearance_ivbuild():
    return _clearance_build()

@pytest.fixture(scope="module")
def clearance_ivbuild_repeat2():
    return _clearance_build(repeat=2)


# Orientation / frame

def test_local_frame_is_right_handed():
    atol = 1e-12
    x, y, z = make_port(30.0, 90, 0).local_frame()
    np.testing.assert_allclose(np.cross(x, y), z, atol=atol)
    np.testing.assert_allclose(np.cross(y, z), x, atol=atol)

def test_local_frame_convention():
    atol = 1e-12
    tor = 30.0
    ph = np.deg2rad(tor)
    radial = [np.cos(ph), np.sin(ph), 0.0]
    toroidal = [-np.sin(ph), np.cos(ph), 0.0]
    np.testing.assert_allclose(
        make_port(tor, 90, 0).orientation, radial, atol=atol
    )
    np.testing.assert_allclose(
        make_port(tor, 90, 90).orientation, toroidal, atol=atol
    )
    np.testing.assert_allclose(
        make_port(tor, 0, 0).orientation, [0, 0, 1], atol=atol
    )
    np.testing.assert_allclose(
        make_port(tor, 180, 0).orientation, [0, 0, -1], atol=atol
    )

def test_same_angles_same_relative_orientation_at_any_toroidal_angle():
    # orientation at tor2 is the orientation at tor1 rotated about z by
    # (tor2 - tor1)
    a, b = make_port(10.0, 70.0, 100.0), make_port(40.0, 70.0, 100.0)
    np.testing.assert_allclose(
        rot_z(30.0) @ a.orientation, b.orientation, atol=1e-12
    )

def test_default_orientation_is_outward_surface_normal():
    p = ivb.Port(TorusSurface(R0, A), 1.0, 30.0, 40.0, radius=10.0, scale=1.0)
    np.testing.assert_allclose(
        p.orientation, surf_normal(40.0, 30.0), atol=1e-3
    )

def test_one_angle_only_rejected():
    with pytest.raises(ValueError):
        ivb.Port(
            TorusSurface(R0, A),
            1.0,
            30.0,
            0.0,
            orientation_polar_angle=90.0,
            radius=10.0,
            scale=1.0
        )

@pytest.mark.parametrize("pol", [0.0, 90.0, 200.0])
def test_base_point_uses_toroidal_and_poloidal_angles(pol):
    p = make_port(30.0, 90.0, 0, pol=pol)
    np.testing.assert_allclose(p.base_point, torus_xyz(pol, 30.0), atol=1e-9)

def test_radius_required_and_positive():
    with pytest.raises(ValueError):
        ivb.Port(TorusSurface(R0, A), 1.0, 30.0, 0.0, scale=1.0)

    with pytest.raises(ValueError):
        ivb.Port(TorusSurface(R0, A), 1.0, 30.0, 0.0, radius=-1.0, scale=1.0)


# Period boundary (unit level)
@pytest.mark.parametrize("tor", [1.0, PERIOD - 1.0], ids=["low", "high"])
def test_base_disk_straddling_boundary_rejected_at_reach_zero(tor):
    p = make_port(tor, 90, 0, radius=20.0)
    with pytest.raises(ValueError, match="period boundary"):
        p.check_within_period(PERIOD, reach=0.0)

@pytest.mark.parametrize("tor", [3.0, PERIOD - 3.0], ids=["low", "high"])
def test_base_disk_just_inside_boundary_accepted(tor):
    p = make_port(tor, 90, 0, radius=20.0)
    p.check_within_period(PERIOD, reach=0.0)

@pytest.mark.parametrize("tor", [0.0, PERIOD], ids=["low", "high"])
def test_base_exactly_on_boundary_rejected(tor):
    """`add_port()` accepts `0 <= tor <= period` inclusive, so a base exactly
    on the edge gets through registration and must be caught here.
    """

    p = make_port(tor, 90, 0, radius=8.0)
    with pytest.raises(ValueError, match="period boundary"):
        p.check_within_period(PERIOD, reach=0.0)

def test_period_above_180_accepts_port_past_180():
    """Base above 200 deg in a 270 deg period. `arctan2()` wraps this to -160.
    """

    p = make_port(200.0, 90, 0, radius=8.0)
    p.check_within_period(270.0, reach=0.0)

def test_period_above_180_still_rejects_straddling_low_edge():
    p = make_port(1.0, 90, 0, radius=20.0)
    with pytest.raises(ValueError, match="period boundary"):
        p.check_within_period(270.0, reach=0.0)

def test_period_above_180_still_rejects_straddling_high_edge():
    p = make_port(269.0, 90, 0, radius=20.0)
    with pytest.raises(ValueError, match="period boundary"):
        p.check_within_period(270.0, reach=0.0)

@pytest.mark.parametrize("tor", [0.0, 100.0, 200.0, 359.0])
def test_full_torus_period_has_no_boundary(tor):
    p = make_port(tor, 90, 0, radius=20.0)
    p.check_within_period(360.0, reach=0.0)

def test_tilted_port_leaving_period_caught_only_by_full_reach():
    p = make_port(40.0, 90, 60.0, pol=0.0, radius=8.0)
    p.inset = p.compute_minimum_inset()
    p.resolved_length = 80.0
    p.check_within_period(PERIOD, reach=0.0)
    with pytest.raises(ValueError, match="period boundary"):
        p.check_within_period(PERIOD)

# Period boundary (build level)
def test_build_rejects_boundary_straddling_port_with_ValueError():
    """End-to-end through `resolve_ports()`. A `RuntimeError` here would
    mean the ray-based length ran before the early check.
    """

    with pytest.raises(ValueError, match="period boundary"):
        build([dict(
            base_toroidal_angle=1.0, base_poloidal_angle=30.0, radius=20.0
        )])

def test_build_accepts_port_near_boundary():
    ivbuild = build([dict(
        base_toroidal_angle=3.0, base_poloidal_angle=30.0, radius=8.0
    )])
    assert ivbuild.ports[0].resolved_length > 0

def test_add_port_rejects_angle_outside_period():
    with pytest.raises(ValueError):
        build([dict(
            base_toroidal_angle=PERIOD + 1.0,
            base_poloidal_angle=30.0,
            radius=8.0
        )])

# Minimum inset
CASES = [
    ("outboard_normal", 0.0, 30.0, lambda n, tp, tt: n),
    ("inboard_normal_saddle", 180.0, 30.0, lambda n, tp, tt: n),
    ("top_normal", 90.0, 30.0, lambda n, tp, tt: n),
    ("outboard_tilt45_poloidal", 0.0, 30.0, lambda n, tp, tt: tilt(n, tp, 45)),
    ("outboard_tilt45_toroidal", 0.0, 30.0, lambda n, tp, tt: tilt(n, tt, 45)),
    ("upper_outboard_tilt30", 45.0, 30.0, lambda n, tp, tt: tilt(n, tt, 30)),
]

def test_outboard_midplane_matches_analytic_curvature():
    """Normal port on the outboard midplane: poloidal curve (1/a) is the
    larger one, so depth ~ a - sqrt(a^2 - r^2).
    """
    port = port_from_axis(surf_normal(0.0, 30.0), 30.0, 0.0)
    analytic = A - np.sqrt(A**2 - RADIUS**2)
    got = port.compute_minimum_inset(min_margin=0.0, margin_frac=0.0)
    assert got == pytest.approx(analytic, rel=0.1)

def test_nearly_flat_surface_gives_margin_only():
    big = TorusSurface(1.0e6, 1.0e5)
    port = ivb.Port(big, 1.0, 30.0, 0.0, radius=RADIUS, scale=1.0)
    assert port.compute_minimum_inset() == pytest.approx(
        0.1 * RADIUS, abs=0.02
    )

@pytest.mark.parametrize(
    "theta,phi,axis_fn", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_inset_matches_brute_force(theta, phi, axis_fn):
    axis = axis_fn(
        surf_normal(theta, phi),
        poloidal_tangent(theta, phi),
        toroidal_tangent(phi)
    )
    port = port_from_axis(axis, phi, theta)
    got = port.compute_minimum_inset(min_margin=0.0, margin_frac=0.0)
    expected = brute_force_behind(phi, theta, axis, RADIUS)

    assert got >= 0.9 * expected - 0.05
    assert got <= 1.25 * expected + 0.05

def test_inward_axis_rejected():
    port = port_from_axis(-surf_normal(0.0, 30.0), 30.0, 0.0)
    with pytest.raises(ValueError, match="outward"):
        port.compute_minimum_inset()

def test_grazing_axis_rejected():
    axis = tilt(surf_normal(0.0, 30.0), poloidal_tangent(0.0, 30.0), 84.0)
    port = port_from_axis(axis, 30.0, 0.0) # cos ~ 0.10 < 0.2
    with pytest.raises(ValueError):
        port.compute_minimum_inset()

def test_footprint_reaching_tangent_region_rejected():
    axis = tilt(surf_normal(0.0, 30.0), poloidal_tangent(0.0, 30.0), 60.0)
    port = port_from_axis(axis, 30.0, 0.0)
    with pytest.raises(ValueError, match="parallel or back-facing"):
        port.compute_minimum_inset()

# Resolve / Length
def test_min_length_matches_total_thickness_on_torus():
    # Tolerance is dominated by the 0.5 cm sagitta floor; tight enough to
    # catch a regression in the ray-trace or curvature estimate
    ivbuild = build([dict(
        base_toroidal_angle=22.5, base_poloidal_angle=30.0, radius=8.0
    )])
    assert ivbuild.ports[0].blanket_length == pytest.approx(
        TOTAL_THICK, abs=0.5
    )

def test_relative_extension():
    ivbuild = build([dict(
        base_toroidal_angle=22.5,
        base_poloidal_angle=30.0,
        radius=8.0,
        extension=0.1,
        relative_extension=True
    )])
    p = ivbuild.ports[0]
    assert p.resolved_length == pytest.approx(1.1 * p.blanket_length)

def test_min_length_overshoots_outer_surface_with_zero_extension():
    ivbuild = _unresolved_build([dict(
        base_toroidal_angle=22.5, base_poloidal_angle=30.0, radius=8.0
    )])
    ivbuild.generate_components()
    p = ivbuild.ports[0]
    outer = ivbuild.Components["shield"]
    end = p.compile_end_points(p.resolved_length)[0]
    assert not outer.isInside(cq.Vector(*(end - 0.05 * p.orientation)))

def test_explicit_length_below_minimum_rejected():
    """`resolve()` rejects any explicit length below the ray-traced minimum
    (true crossing is `TOTAL_THICK`; `resolve()` adds a sagitta allowance).
    """

    ivbuild = _unresolved_build([dict(
        base_toroidal_angle=22.5,
        base_poloidal_angle=30.0,
        radius=8.0,
        length=TOTAL_THICK - 2.0
    )])
    with pytest.raises(ValueError, match="shorter than the minimum required"):
        ivbuild.generate_components()

def test_explicit_length_just_above_minimum_accepted():
    """Other side of the threshold: just above the `resolve()` minimum
    (~`TOTAL_THICK` + sagitta allowance), the build must succeed.
    """
    ivbuild = _unresolved_build([dict(
        base_toroidal_angle=22.5,
        base_poloidal_angle=30.0,
        radius=8.0,
        length=TOTAL_THICK + 0.5
    )])
    ivbuild.generate_components()

def test_resolve_error_reports_offending_port_index():
    ivbuild = _unresolved_build(
        [
            dict(
                base_toroidal_angle=11.0,
                base_poloidal_angle=30.0,
                radius=8.0,
                length=TOTAL_THICK + 5.0,
            ),
            dict(
                base_toroidal_angle=34.0,
                base_poloidal_angle=30.0,
                radius=8.0,
                length=0.5 * TOTAL_THICK,
            ),
        ]
    )
    with pytest.raises(ValueError, match=r"Port 1: "):
        ivbuild.generate_components()

def test_check_port_exits_backstop_catches_blind_hole():
    ivbuild = _unresolved_build(
        [dict(base_toroidal_angle=22.5, base_poloidal_angle=30.0, radius=8.0)]
    )
    p = ivbuild.ports[0]
    p.inset = p.compute_minimum_inset()
    outer = list(ivbuild.Surfaces.values())[-1].generate_surface()

    p.resolved_length = 0.5 * TOTAL_THICK
    with pytest.raises(ValueError, match="Port 0 does not fully exit"):
        ivbuild._check_port_exits(outer)

    p.resolved_length = TOTAL_THICK + 5.0
    ivbuild._check_port_exits(outer)

def test_generate_port_is_valid_cylinder():
    radius = 8.0
    p = make_port(22.5, 90, 0, pol=30.0, radius=radius)
    p.inset = p.compute_minimum_inset()
    p.resolved_length = 80.0
    solid = p.generate_port()
    assert solid.isValid()
    assert solid.Volume() == pytest.approx(
        np.pi * radius**2 * (80.0 + p.inset), rel=1e-6
    )

# CAD Booleans
def test_inset_is_sufficient_and_necessary_on_a_shell():
    t, length = 30.0, 100.0
    port = port_from_axis(surf_normal(0.0, 30.0), 30.0, 0.0)
    base, n = port.base_point, port.orientation
    inset = port.compute_minimum_inset()
    shell = cq.Solid.makeTorus(R0, A + t).cut(cq.Solid.makeTorus(R0, A))

    def removed(inset_val):
        cyl = cq.Solid.makeCylinder(
            RADIUS,
            length + inset_val,
            cq.Vector(*(base - inset_val * n)),
            cq.Vector(*n)
        )
        cut = shell.cut(cyl)
        assert cut.isValid()
        return shell.Volume() - cut.Volume()
    
    v_computed = removed(inset)
    v_generous = removed(inset + 50.0)
    v_zero = removed(0.0)
    assert v_computed == pytest.approx(v_generous, rel=1e-4)
    assert v_zero < v_computed * (1.0 - 1e-3)

def test_ports_cut_in_every_repeated_period():
    ivbuild = build(
        [dict(
            base_toroidal_angle=22.5,
            base_poloidal_angle=30.0,
            radius=8.0,
            extension=0.1,
            relative_extension=True,
        )],
        repeat=2,
        resolve=False,
    )
    ivbuild.generate_components()

    port = ivbuild.ports[0]
    blanket = ivbuild.Components["blanket"]
    u, _ = port._section_axes()
    on_axis = port.base_point + 20.0 * port.orientation
    off_axis = on_axis + 3.0 * port.radius * u

    for k in range(ivbuild._repeat + 1):
        R = rot_z(k * PERIOD)
        assert not blanket.isInside(cq.Vector(*(R @ on_axis))), f"no hole, k={k}"
        assert blanket.isInside(cq.Vector(*(R @ off_axis))), f"no solid, k={k}"

# Magnet clearance
def test_magnet_inside_port_exact_overlap(clearance_ivbuild):
    ivbuild = clearance_ivbuild
    magnet = _box(_on_axis(ivbuild.ports[0]))
    conflicts = ivbuild.check_port_magnet_clearance(
        [magnet], on_conflict="warn"
    )

    assert len(conflicts) == 1
    p_idx, k, m_idx, vol = conflicts[0]
    assert (p_idx, k, m_idx) == (0, 0, 0)
    assert vol == pytest.approx(4.0**3, rel=1e-3)

def test_conflict_raises_with_readable_message(clearance_ivbuild):
    ivbuild = clearance_ivbuild
    magnet = _box(_on_axis(ivbuild.ports[0]))
    with pytest.raises(ValueError, match=r"Port/magnet overlaps*\(0, 0, 0, "):
        ivbuild.check_port_magnet_clearance([magnet])

def test_bbox_overlap_without_volume_overlap_is_not_a_conflict(
    clearance_ivbuild
):
    ivbuild = clearance_ivbuild
    port = ivbuild.ports[0]
    bb = port.generate_port().BoundingBox()
    center = np.array([bb.xmax, bb.ymax, bb.xmax]) - 1.5
    assert not port.generate_port().isInside(cq.Vector(*center))
    assert ivbuild.check_port_magnet_clearance([_box(center, side=2.0)]) == []

@pytest.mark.parametrize(
    "clearance,expect_conflict", [(0.0, False), (3.0, True)]
)
def test_clearance_widens_the_keep_out(
    clearance_ivbuild, clearance, expect_conflict
):
    ivbuild = clearance_ivbuild
    port = ivbuild.ports[0]
    u, _ = port._section_axes()
    magnet = _box(_on_axis(port) + 10.0 * u, side=2.0)

    conflicts = ivbuild.check_port_magnet_clearance(
        [magnet], clearance=clearance, on_conflict="warn"
    )
    assert bool(conflicts) is expect_conflict

def test_conflict_found_in_repeated_period_only(clearance_ivbuild_repeat2):
    ivbuild = clearance_ivbuild_repeat2
    port = ivbuild.ports[0]
    magnet = _box(rot_z(2 * PERIOD) @ _on_axis(port))
    conflicts = ivbuild.check_port_magnet_clearance(
        [magnet], on_conflict="warn"
    )
    assert [(c[0], c[1], c[2]) for c in conflicts] == [(0, 2, 0)]
    assert conflicts[0][3] == pytest.approx(4.0**3, rel=1e-3)

def test_reports_correct_magnet_index_and_ignores_distant_magnet(
    clearance_ivbuild
):
    ivbuild = clearance_ivbuild
    far = _box([0.0, 0.0, 300.0])
    near = _box(_on_axis(ivbuild.ports[0]))
    conflicts = ivbuild.check_port_magnet_clearance(
        [far, near], on_conflict="warn"
    )
    assert [(c[0], c[1], c[2]) for c in conflicts] == [(0, 0, 1)]