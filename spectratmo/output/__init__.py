"""Output
=========

.. autosummary::
   :toctree:

   spatial
   global_tmean
   energy_budget
   spectra
   seb

"""

import os

from h5netcdf import File
# we should try not to use the legacy API...
import h5netcdf.legacyapi as net

from spectratmo.phys_const import khi

from spectratmo.planets import earth

import numpy as np

class BaseOutput(object):
    _name = 'base'

    def __init__(self, dataset):
        self._dataset = dataset
        #self.names=["E_A","E_K","C","F_A","T_Av","T_Ah","T_Kh","T_Kv","T_Khrot","F_Kt","F_Kv","Lcori","Lcalt"]

        ds = self._dataset

        lat = ds.lats
        lon = ds.lons
        nlat = lat.size
        nlon = lon.size

        dlat_np = abs(90 - ( (lat[0]+lat[1])/2.0 ))
        dlat_sp = abs(-90 - ( (lat[nlat-1]+lat[nlat-2])/2.0 ))

        dlat = np.zeros(nlat)
        dlat[0]      = dlat_np
        dlat[nlat-1] = dlat_sp
        for i in range (1, nlat-1):
            dlat[i] = (lat[i-1]+lat[i])/2.0 - (lat[i]+lat[i+1])/2.0
        if dlat[3] < 0:
            dlat[0]      = - dlat[0]
            dlat[nlat-1] = - dlat[nlat-1]
        self.dlat_rad = dlat / 180.0 * np.pi

        dlam = lon[0] - lon[1]
        self.dlam_rad = dlam / 180.0 * np.pi
        #dlam_rad_int = np.absolute(dlam_rad)

    def dlam(self,FX):

        ds = self._dataset
        lat = ds.lats
        nlat = lat.size
        nlon = ds.lons.size

        dFXdlam = np.empty_like(FX)

        dFXdlam[:,0] = (FX[:,0]-FX[:,1])/self.dlam_rad
        dFXdlam[:,nlon-1] = (FX[:,nlon-2]-FX[:,nlon-1])/self.dlam_rad
        for l in range(1,nlon-1):
            dFXdlam[:,l] = (FX[:,l-1]-FX[:,l+1])/(2.0*self.dlam_rad)

        return dFXdlam / np.reshape((earth.radius*np.cos(np.pi/180.0*lat[:])),(nlat,1))

    def dphi(self,FX):

        ds = self._dataset
        nlat = ds.lats.size
        nlon = ds.lons.size

        dFXdphi = np.empty_like(FX)
        dFXdphi[0,:] = (FX[0,:]-FX[1,:])/self.dlat_rad[0]
        dFXdphi[nlat-1,:]=(FX[nlat-2,:]-FX[nlat-1,:])/self.dlat_rad[nlat-1]
        for i in range(1,nlat-1):
            dFXdphi[i,:] = (FX[i-1,:]-FX[i+1,:])/(2.0*self.dlat_rad[i])

        return dFXdphi / earth.radius


    def compute_loopallinstants(self):
        for instant in self._dataset.instants:
            self.compute_1instant(instant)

    def _get_path_from_name(self, name):
        path = os.path.join(self._dataset.path, name + '.nc')
        if not path.endswith('.nc'):
            path += '.nc'
        return path

    def check_if_file_exists(self, name):
        path = self._get_path_from_name(name)
        if os.path.isfile(path):
            return True
        else:
            return False

    def get_spat1d(self, varname='save'):
        fname = self._get_path_from_name(varname)
        with File(fname) as fx:
            arr = fx[varname][:]
        return arr

    def save_spat3d(self, var, varname='save'):
        """Save

        We have to improve this function:
        save_spat3d(self, var, varname, filename=None)

        Also, now it deletes any existing file !

        """
        ds = self._dataset
        fname = self._get_path_from_name(varname)

        # PA: I think it would be nicer to use the new API of
        # h5netcdf (i.e. as in get_netcdf_spat1d).
        # see https://pypi.python.org/pypi/h5netcdf/
        # FTV: I agree, but the website said it was experimental 
        # and "not yet finished" so I'm not so sure how much we 
        # can trust it at this point. 
        with net.Dataset(fname, 'w') as dsn:
            dsn.createDimension('lev', ds.nlev)
            dsn.createDimension('lat', self.nlat)
            dsn.createDimension('lon', self.nlon)
            levs = dsn.createVariable('lev', float, ('lev',))
            levs[:] = ds.pressure_levels
            lats = dsn.createVariable('lat', float, ('lat',))
            lats[:] = ds.lats
            lons = dsn.createVariable('lon', float, ('lon',))
            lons[:] = ds.lons
            v = dsn.createVariable(varname, float, ('lev', 'lat', 'lon',))
            v[:] = var

    def save_spat1d(self, var, varname='save'):
        """Save

        We have to improve this function:
        save_spat1d(self, var, varname, filename=None)

        Also, now it deletes any existing file !

        """
        ds = self._dataset
        fname = self._get_path_from_name(varname)
        #print fname
        # with new API?
        with net.Dataset(fname, 'w') as dsn:
            dsn.createDimension('lev', ds.nlev)
            levs = dsn.createVariable('lev', float, ('lev',))
            levs[:] = ds.pressure_levels
            v = dsn.createVariable(varname, float, ('lev',))
            v[:] = var

    def calculate_Lambda(self):
        ds = self._dataset
        #print 'calc Lambda'
        #print earth.pressure_m
        #print ds.pressure_levels[:]
        #print khi
        #print (earth.pressure_m/ds.pressure_levels[:])**khi
        #print earth
        #print earth.radius
        #print earth.pressure_m
        
        return (earth.pressure_m/ds.pressure_levels[:])**khi

    def compute_hmean(self, f_xy):
        """Compute the horizontal mean."""
        ds = self._dataset
        delta = 360./ds.nlon
        return (delta*np.pi/180)**2 / (4*np.pi) * np.sum(f_xy*ds.oper.cosLATS)

    def compute_hmean_representative(self, f_xy, ip=None, beta=None):
        """Compute the horizontal "representative" mean."""
        ds=self._dataset
        if beta is None:
            if not hasattr(self, 'beta3d'):
                if ds.global_tmean.with_beta:
                    raise ValueError(
                        'first run dataset.compute_tmean_ps_beta().')
                else:
                    result = self.compute_hmean(f_xy)
            else:
                if ip < self.nb_lev_beta:
                    result = self.compute_hmean(
                        f_xy*self.beta3d[ip])/self.beta_mean1d[ip]
                elif ip >= self.nb_lev_beta:
                    result = self.compute_hmean(f_xy)
        else:
            result = self.compute_hmean(f_xy*beta)/self.compute_hmean(beta)
        return result

    def partialp_f(self, f):
        """Compute derivative over pressure."""
        ds = self._dataset
        p = ds.pressure_levels
        npx = p.size-2

        result = np.empty_like(f)
        result[1:-1] = (f[2:] - f[:-2]) / np.reshape((p[2:] - p[:-2]),(npx,1,1))
        result[0] = (f[1] - f[0]) / np.reshape((p[1] - p[0]),(1,1,1))
        result[-1] = (f[-1] - f[-2]) / np.reshape((p[-1] - p[-2]),(1,1,1))
        return result

