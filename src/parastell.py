import numpy as np
import cadquery as cq

m2cm = 100

class Stellarator(object):
    def __init__():
        pass

class StellaratorMagnetSet():
    def __init__():
        pass
    
class StellaratorMagnetCoil():
    def __init__():
        pass

class StellaratorMagnetFilament(object);
    def __init__():
        pass

class StellaratorInvesselComponent(object):
    def __init__():
        pass

class StellaratorSurface(object):
    '''
    An object that represents a surface formed by lofting across a number of
    "ribs" located at different toroidal plans and offset from a reference surface.

    Args:
        vmec : a vmec object from pystell_uw
        s (float) : the flux surface coordinate defining the point of reference for offset
        offset (np.array(double)) : for each poloidal angle and toroidal, an offset in [cm] from the 
            surface defined by s
        theta (np.array(double)) : the set of poloidal angles in [rad]
        phi (np.array(double)) : the set of toroidal angles for this rib in [rad]
        scale (double) : a scaling factor between the units of vmec and [cm] (default: m2cm = 100)
    '''
    def __init__(
            self,
            vmec_,
            s_,
            offset_,
            theta_,
            phi_,
            scale_ = m2cm
    ):
        self.vmec = vmec_
        self.s = s_
        self.offset = offset_
        self.theta = theta_
        self.phi = phi_
        self.scale = scale_

        self.ribs = [ 
            StellaratorRib(self.vmec, self.s, self.offset[i,:], self.theta, phi, self.scale)
            for i, phi in enumerate(self.phi) ]

        ribs = [rib.rib for rib in self.ribs]

        # loft over ribs

class StellaratorRib(object):
    '''
    An object that represents a curve formed from different poloidal points
    in a single toroidal plane.

    Args:
        vmec : a vmec object from pystell_uw
        s (float) : the flux surface coordinate defining the point of reference for offset
        offset (np.array(double)) : for each poloidal angle, an offset in [cm] from the 
            surface defined by s
        theta (np.array(double)) : the set of poloidal angles in [rad]
        phi (double) : the toroidal angle for this rib in [rad]
        scale (double) : a scaling factor between the units of vmec and [cm] (default: m2cm = 100)
    '''
    def __init__(
            self,
            vmec_,
            s_,
            offset_,
            theta_,
            phi_,
            scale_ = m2cm
    ):
        self.vmec = vmec_
        self.s = s_
        self.offset = offset_
        self.theta = theta_
        self.phi = phi_
        self.scale = scale_

        if not np.all(self.offset >= 0):
            raise ValueError(
                'Offset must be greater than or equal to 0. Check thickness inputs '
                'for negative values'
            )

        self.r_loci = self.vmec2xyz()

        if self.offset > 0:
            self.r_loci += (self.offset.T * self.surf_norm().T).T

        self.r_loci = np.append(self.r_loci, self.r_loci[0,:])

        self.rib = cq.Workplane("XY").spline(self.r_loci).close()

    def vmec2xyz(self, poloidal_offset=0):
        '''
        Return an N x 3 NumPy array containing the Cartesian coordinates of the points
        at this toroidal angle and N different poloidal angles, each offset slightly

        args:
            poloidal_offset (double) : some offset to apply to the full set of poloidal
                angles for evaluating the location of the Cartesian points (default: 0)
        '''
        return np.array([self.scale * self.vmec.vmec2xyz(self.s, theta, self.phi) 
                         for theta in (self.theta + poloidal_offset)])

    def normalize(vec_list):
        return np.divide(vec_list.T, np.linalg.norm(vec_list, axis=1).T).T

    def surf_norm(self):
        '''
        Approximate the normal to the curve at each poloidal angle by first approximating
        the tangent to the curve and then taking the cross-product of that tangent with
        a vector defined as normal to the plane at this toroidal angle
        '''

        eps = 1e-4
        next_pt_loci = self.vmec2xyz(eps)

        tangent = next_pt_loci - self.r_loci

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        norm = np.cross(self.plane_norm, tangent)
        return normalize(norm)
    

def parse_args():
    pass

def parastell():
    args = parse_args()
    pass

if __name__ == "__main__":
    parastell()