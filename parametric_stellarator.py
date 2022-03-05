import cadquery as cq
from flux_surface import flux_surface
import read_vmec


def parametric_stellarator(plas_eq, norm_tf):
    """Generates STEP files for components of a parametrically-defined
    stellarator based on user-supplied plasma equilibrium VMEC data
    using CadQuery. Currently, this package generates STEP files only
    for user-specified closed flux surfaces.

    Arguments:
        plas_eq (str): path to plasma equilibrium NetCDF file
        norm_tf (float or iterable of floats): normalized toroidal
            flux(es) defining closed flux surface(s).
    """
    
    # Load plasma equilibrium data
    vmec = read_vmec.vmec_data(plas_eq)

    # Generate flux surface(s)
    cfs = flux_surface(vmec, norm_tf)

    # Export flux surface(s) as STEP file(s)
    for i, surf in enumerate(cfs, 1):
        cq.exporters.export(surf, 'cfs%d.step' % (i))
