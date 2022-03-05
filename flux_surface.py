import cadquery as cq
from numpy import pi


def flux_surface(vmec, norm_tf):
    """Creates plasma closed flux surface(s) as CadQuery object(s)
    based on VMEC data.

    Arguments:
        vmec (object): read_vmec.vmec_data class object
        norm_tf (float or iterable of floats): normalized toroidal
            flux(es) defining closed flux surface(s).

    Returns:
        cfs (iterable of object(s)): closed flux surface CadQuery
            object(s)
    """

    # Ensure norm_tf is in list format
    if type(norm_tf) is not list:
        norm_tf = [norm_tf]
    
    # Initialize list of flux surface(s)
    cfs = list()

    # Generate flux surface(s) and append to flux surface list
    for flux in norm_tf:
        surf = cq.Workplane('XY').parametricSurface(lambda u, v:
            vmec.vmec2xyz(flux, u*2*pi, v*2*pi))
        cfs.append(surf)

    return cfs
