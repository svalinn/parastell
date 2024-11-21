from pathlib import Path
import os
import inspect
import subprocess

import cubit

initialized = False


def init_cubit():
    """Initializes Coreform Cubit with the DAGMC plugin."""
    global initialized

    if not initialized:
        cubit_plugin_dir = Path(
            os.path.dirname(inspect.getfile(cubit))
        ) / Path("plugins")
        cubit.init(
            [
                "cubit",
                "-nojournal",
                "-nographics",
                "-information",
                "off",
                "-warning",
                "off",
                "-commandplugindir",
                str(cubit_plugin_dir),
            ]
        )
        initialized = True


def import_step_cubit(filename, import_dir):
    """Imports STEP file into Coreform Cubit.

    Arguments:
        filename (str): name of STEP input file, excluding '.step' extension.
        import_dir (str): directory from which to import STEP file.

    Returns:
        vol_id (int): Cubit volume ID of imported CAD solid.
    """
    init_cubit()

    import_path = Path(import_dir) / Path(filename).with_suffix(".step")
    cubit.cmd(f'import step "{import_path}" heal')
    vol_id = cubit.get_last_id("volume")

    return vol_id


def export_step_cubit(filename, export_dir=""):
    """Export CAD solid as a STEP file via Coreform Cubit.

    Arguments:
        filename (str): name of STEP output file, excluding '.step' extension.
        export_dir (str): directory to which to export the STEP output file
            (defaults to empty string).
    """
    init_cubit()

    export_path = Path(export_dir) / Path(filename).with_suffix(".step")
    cubit.cmd(f'export step "{export_path}" overwrite')


def export_cub5(filename, export_dir=""):
    """Export cub5 representation of model (native Cubit format).

    Arguments:
        filename (str): name of cub5 output file, excluding '.cub5' extension.
        export_dir (str): directory to which to export the cub5 output file
            (defaults to empty string).
    """
    init_cubit()

    export_path = Path(export_dir) / Path(filename).with_suffix(".cub5")
    cubit.cmd(f'save cub5 "{export_path}" overwrite')


def export_mesh_cubit(filename, export_dir="", delete_upon_export=True):
    """Exports Cubit mesh to H5M file format, first exporting to Exodus format
    via Coreform Cubit and converting to H5M via MOAB.

    Arguments:
        filename (str): name of H5M output file, excluding '.h5m' extension.
        export_dir (str): directory to which to export the H5M output file
            (defaults to empty string).
        delete_upon_export (bool): delete the mesh from the Cubit instance
            after exporting. Prevents inclusion of mesh in future exports.
    """
    init_cubit()

    exo_path = Path(export_dir) / Path(filename).with_suffix(".exo")
    h5m_path = Path(export_dir) / Path(filename).with_suffix(".h5m")

    cubit.cmd(f'export mesh "{exo_path}" overwrite')
    subprocess.run(f"mbconvert {exo_path} {h5m_path}", shell=True)
    Path.unlink(exo_path)

    # Delete any meshes present to prevent inclusion in future Cubit mesh
    # exports
    if delete_upon_export:
        cubit.cmd(f"delete mesh volume all propagate")


def export_dagmc_cubit_legacy(
    faceting_tolerance=None,
    length_tolerance=None,
    normal_tolerance=None,
    filename="dagmc",
    export_dir="",
):
    """Exports DAGMC neutronics H5M file of ParaStell components via legacy
    plug-in faceting method for Coreform Cubit.

    Arguments:
        faceting_tolerance (float): maximum distance a facet may be from
            surface of CAD representation for DAGMC export (defaults to None).
        length_tolerance (float): maximum length of facet edge for DAGMC export
            (defaults to None).
        normal_tolerance (float): maximum change in angle between normal vector
            of adjacent facets (defaults to None).
        filename (str): name of DAGMC output file, excluding '.h5m' extension
            (defaults to 'dagmc').
        export_dir (str): directory to which to export the DAGMC output file
            (defaults to empty string).
    """
    init_cubit()

    tol_str = ""

    if faceting_tolerance is not None:
        tol_str += f"faceting_tolerance {faceting_tolerance}"
    if length_tolerance is not None:
        tol_str += f"length_tolerance {length_tolerance}"
    if normal_tolerance is not None:
        tol_str += f"normal_tolerance {normal_tolerance}"

    export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")
    cubit.cmd(f'export dagmc "{export_path}" {tol_str} make_watertight')


def tag_surface_legacy(surface_id, tag):
    """Applies a boundary condition to a surface in cubit following UWUW
    syntax.

    Arguments:
        surface_id (int): Surface to tag
        tag (str): boundary type
    """
    cubit.cmd(f'group "boundary:{tag}" add surf {surface_id}')


def tag_surface_native(surface_id, tag):
    """Applies a boundary condition to a surface in cubit following the
        native coreform syntax

    Arguments:
        surface_id (int): Surface to tag
        tag (str): boundary type
    """
    cubit.cmd(f"create sideset {surface_id}")
    cubit.cmd(f"sideset {surface_id} name 'boundary:{tag}'")
    cubit.cmd(f"sideset {surface_id} add surface {surface_id}")


def export_dagmc_cubit_native(
    anisotropic_ratio=100.0,
    deviation_angle=5.0,
    filename="dagmc",
    export_dir="",
    delete_upon_export=True,
):
    """Exports DAGMC neutronics H5M file of ParaStell components via native
    faceting method for Coreform Cubit.

    Arguments:
        anisotropic_ratio (float): controls edge length ratio of elements
            (defaults to 100.0).
        deviation_angle (float): controls deviation angle of facet from surface
            (i.e., lesser deviation angle results in more elements in areas
            with higher curvature) (defaults to 5.0).
        filename (str): name of DAGMC output file, excluding '.h5m' extension
            (defaults to 'dagmc').
        export_dir (str): directory to which to export the DAGMC output file
            (defaults to empty string).
        delete_upon_export (bool): delete the mesh from the Cubit instance
            after exporting. Prevents inclusion of mesh in future exports.
    """
    init_cubit()

    cubit.cmd(
        f"set trimesher coarse on ratio {anisotropic_ratio} "
        f"angle {deviation_angle}"
    )
    cubit.cmd("surface all scheme trimesh")
    cubit.cmd("mesh surface all")

    export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")
    cubit.cmd(f'export cf_dagmc "{export_path}" overwrite')

    # Delete any meshes present to prevent inclusion in future Cubit mesh
    # exports
    if delete_upon_export:
        cubit.cmd(f"delete mesh volume all propagate")


def create_new_cubit_instance():
    """Creates new cubit instance checking if Cubit has been already
    initialized.

    """
    if initialized:
        cubit.cmd("new")
    else:
        init_cubit()
