"""Spectra
==========


"""

from spectratmo.output import BaseOutput
from spectratmo.phys_const import R
import numpy as np

class EnergyBudget(BaseOutput):

    _name = 'energy_budget'

    def compute_1instant(self, instant):
        r"""Compute energy budget

        Notes
        -----

        .. math::

           E_K =

           E_A =

        """
        ds = self._dataset

        Lambda_p = self.calculate_Lambda()

        u3d = ds.get_spatial3dvar('u', instant)
        v3d = ds.get_spatial3dvar('v', instant)
        o3d = ds.get_spatial3dvar('w', instant)
        T3d = ds.get_spatial3dvar('t', instant)

        gamma_p = ds.global_tmean.compute_gamma()

        E_Kp = np.zeros(ds.nlev)
        E_Pp = np.zeros(ds.nlev)
        Cp   = np.zeros(ds.nlev)

        for ip, p in enumerate(ds.pressure_levels):
            gamma = gamma_p[ip]
            Lambda = Lambda_p[ip] 

            u2d = u3d[ip]
            v2d = v3d[ip]
            o2d = o3d[ip]
            T2d = T3d[ip]

            theta2d = Lambda * T2d
            theta_hmean = self.compute_hmean_representative(theta2d, ip)
            theta2d_dev = theta2d - theta_hmean

            alpha2d = R / (p) * T2d

            E_K = self.compute_hmean_representative(
                u2d**2 + v2d**2, ip)
            E_Kp[ip] = E_K

            E_P = gamma * self.compute_hmean_representative(
                theta2d_dev**2, ip) /2
            E_Pp[ip] = E_P

            C = self.compute_hmean_representative(
                o2d * alpha2d, ip)
            Cp[ip] = C

        return E_Kp,E_Pp,Cp

    def compute_tmean(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
