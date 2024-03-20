import cubit
from pathlib import Path
import os
import inspect
import subprocess


def init_cubit(cubit_flag):
    """Initializes Coreform Cubit with the DAGMC plugin.

    Arguments:
        cubit_flag (bool): flag to indicate whether Coreform Cubit has been
            initialized.

    Returns:
        cubit_flag (bool): flag to indicate whether Coreform Cubit has been
            initialized.
    """
    if not cubit_flag:
        cubit_plugin_dir = (
            Path(os.path.dirname(inspect.getfile(cubit))) / Path('plugins')
        )
        cubit.init([
            'cubit',
            '-nojournal',
            '-nographics',
            '-information', 'off',
            '-warning', 'off',
            '-commandplugindir',
            str(cubit_plugin_dir)
        ])
        cubit_flag = True

    return cubit_flag


def import_step_cubit(filename, import_dir):
    """Imports STEP file into Coreform Cubit.

    Arguments:
        filename (str): name of STEP input file, excluding '.step' extension.
        import_dir (str): directory from which to import STEP file.

    Returns:
        vol_id (int): Cubit volume ID of imported CAD solid.
    """
    import_path = Path(import_dir) / Path(filename).with_suffix('.step')
    cubit.cmd(f'import step "{import_path}" heal')
    vol_id = cubit.get_last_id("volume")

    return vol_id


def export_step_cubit(filename, export_dir=''):
    """Export CAD solid as a STEP file via Coreform Cubit.

    Arguments:
        filename (str): name of STEP output file, excluding '.step' extension.
        export_dir (str): directory to which to export the STEP output file
            (defaults to empty string).
    """
    export_path = Path(export_dir) / Path(filename).with_suffix('.step')
    cubit.cmd(f'export step "{export_path}" overwrite')


def export_tet_mesh_cubit(volume_ids, filename, export_dir=''):
    """Creates tetrahedral mesh of CAD volumes and exports to Exodus format via
    Coreform Cubit and converts to H5M via MOAB.

    Arguments:
        volume_ids (iterable of int): iterable of Cubit volume IDs.
        filename (str): name of H5M output file, excluding '.h5m' extension.
        export_dir (str): directory to which to export the H5M output file
            (defaults to empty string).
    """
    for vol in volume_ids:
        cubit.cmd(f'volume {vol} scheme tetmesh')
        cubit.cmd(f'mesh volume {vol}')

    exo_path = Path(export_dir) / Path(filename).with_suffix('.exo')
    h5m_path = Path(export_dir) / Path(filename).with_suffix('.h5m')

    cubit.cmd(f'export mesh "{exo_path}" overwrite')
    subprocess.run(f'mbconvert {exo_path} {h5m_path}', shell=True)
    Path.unlink(exo_path)


def orient_spline_surfaces(volume_id):
    """Extracts the inner and outer surface IDs for a given ParaStell in-vessel
    component volume in Coreform Cubit.

    Arguments:
        volume_id (int): Cubit volume ID.

    Returns:
        inner_surface_id (int): Cubit ID of in-vessel component inner surface.
        outer_surface_id (int): Cubit ID of in-vessel component outer surface.
    """
    surfaces = cubit.get_relatives('volume', volume_id, 'surface')

    spline_surfaces = []
    for surface in surfaces:
        if cubit.get_surface_type(surface) == 'spline surface':
            spline_surfaces.append(surface)

    if len(spline_surfaces) == 1:
        outer_surface_id = spline_surfaces[0]
        inner_surface_id = None
    else:
        # The outer surface bounding box will have the larger maximum XY value
        if (
            cubit.get_bounding_box('surface', spline_surfaces[1])[4] >
            cubit.get_bounding_box('surface', spline_surfaces[0])[4]
        ):
            outer_surface_id = spline_surfaces[1]
            inner_surface_id = spline_surfaces[0]
        else:
            outer_surface_id = spline_surfaces[0]
            inner_surface_id = spline_surfaces[1]

    return inner_surface_id, outer_surface_id


def merge_layer_surfaces(components_dict):
    """Merges ParaStell in-vessel component surfaces in Coreform Cubit based on
    surface IDs rather than imprinting and merging all. Assumes that the
    components dictionary is ordered radially outward. Note that overlaps
    between magnet volumes and in-vessel components will not be merged in this
    workflow.

    Arguments:
        components_dict (dict): dictionary of ParaStell components. This
            dictionary must have the form
            {
                'component_name': {
                    'vol_id': Coreform Cubit volume ID(s) for component (int or
                        iterable of int)
                    (additional keys are allowed)
                }
            }
    """
    # Tracks the surface id of the outer surface of the previous layer
    prev_outer_surface_id = None

    for component in components_dict.keys():
        vol_id = component['vol_id']
        # Skip merging for magnets
        if len(vol_id) > 1:
            continue

        inner_surface_id, outer_surface_id = orient_spline_surfaces(vol_id)

        # Conditionally skip merging (first iteration only)
        if prev_outer_surface_id is None:
            prev_outer_surface_id = outer_surface_id
        else:
            cubit.cmd(
                f'merge surface {inner_surface_id} {prev_outer_surface_id}'
            )
            prev_outer_surface_id = outer_surface_id


def export_h5m_cubit_legacy(
    components_dict, filename='dagmc', export_dir='', skip_imprint=False,
    faceting_tolerance=None, length_tolerance=None, normal_tolerance=None
):
    """Exports DAGMC neutronics H5M file of ParaStell components via legacy
    plug-in faceting method for Coreform Cubit.

    Arguments:
        components_dict (dict): dictionary of ParaStell components. This
            dictionary must have the form
            {
                'component_name': {
                    'vol_id': Coreform Cubit volume ID(s) for component (int or
                        iterable of int)
                    'h5m_tag': material tag for component (str)
                    (additional keys are allowed)
                }
            }
        filename (str): name of H5M output file, excluding '.step' extension
            (defaults to 'dagmc').
        export_dir (str): directory to which to export the H5M output file
            (defaults to empty string).
        skip_imprint (bool): flag for whether the export routine should skip
            the imprint procedure (defaults to False).
        faceting_tolerance (double): maximum distance a facet may be from
            surface of CAD representation for H5M export (defaults to None).
        length_tolerance (double): maximum length of facet edge for H5M export
            (double, defaults to None).
        normal_tolerance (double): maximum change in angle between normal
            vector of adjacent facets (defaults to None).
    """
    if skip_imprint:
        merge_layer_surfaces()

    else:
        cubit.cmd('imprint volume all')
        cubit.cmd('merge volume all')

    for component in components_dict.keys():
        if len(component['vol_id']) > 1:
            vol_id_str = " ".join(str(i) for i in component["vol_id"])
        else:
            vol_id_str = str(component['vol_id'])

        cubit.cmd(
            f'group "mat:{component["h5m_tag"]}" add volume {vol_id_str}'
        )

    facet_tol_str = ''
    len_tol_str = ''
    norm_tol_str = ''

    if faceting_tolerance is not None:
        facet_tol_str = f'faceting_tolerance {faceting_tolerance}'
    if length_tolerance is not None:
        len_tol_str = f'length_tolerance {length_tolerance}'
    if normal_tolerance is not None:
        norm_tol_str = f'normal_tolerance {normal_tolerance}'

    export_path = Path(export_dir) / Path(filename).with_suffix('.h5m')
    cubit.cmd(
        f'export dagmc "{export_path}" {facet_tol_str} {len_tol_str} '
        f'{norm_tol_str} make_watertight'
    )


def export_h5m_cubit_native(
    components_dict, filename='dagmc', export_dir='', skip_imprint=False,
    anisotropic_ratio=100.0, deviation_angle=5.0
):
    """Exports DAGMC neutronics H5M file of ParaStell components via native faceting method for Coreform Cubit.

    Arguments:
        components_dict (dict): dictionary of ParaStell components. This
            dictionary must have the form
            {
                'component_name': {
                    'vol_id': Coreform Cubit volume ID(s) for component (int or
                        iterable of int)
                    'h5m_tag': material tag for component (str)
                    (additional keys are allowed)
                }
            }
        filename (str): name of H5M output file, excluding '.step' extension
            (defaults to 'dagmc').
        export_dir (str): directory to which to export the H5M output file
            (defaults to empty string).
        skip_imprint (bool): flag for whether the export routine should skip
            the imprint procedure (defaults to False).
        anisotropic_ratio (double): controls edge length ratio of elements
            (defaults to 100.0).
        deviation_angle (double): controls deviation angle of facet from
            surface (i.e., lesser deviation angle results in more elements in
            areas with higher curvature) (defaults to 5.0).
    """
    if skip_imprint:
        merge_layer_surfaces()
    else:
        cubit.cmd('imprint volume all')
        cubit.cmd('merge volume all')

    cubit.cmd('set duplicate block elements off')

    for component in components_dict.keys():
        if len(component['vol_id']) > 1:
            block_id = min(component['vol_id'])
            vol_id_str = " ".join(str(i) for i in component["vol_id"])
        else:
            block_id = component['vol_id']
            vol_id_str = str(component['vol_id'])

        cubit.cmd(
            f'create material "{component["h5m_tag"]}" property_group '
            '"CUBIT-ABAQUS"'
        )
        cubit.cmd(
            f'block {block_id} add volume {vol_id_str}'
        )
        cubit.cmd(
            f'block {block_id} material \'{component['h5m_tag']}\''
        )

    cubit.cmd(
        f'set trimesher coarse on ratio {anisotropic_ratio} '
        f'angle {deviation_angle}'
    )
    cubit.cmd("surface all scheme trimesh")
    cubit.cmd("mesh surface all")

    export_path = Path(export_dir) / Path(filename).with_suffix('.h5m')
    cubit.cmd(f'export cf_dagmc "{export_path}" overwrite')
