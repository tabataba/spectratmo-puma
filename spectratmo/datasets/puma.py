"""PUMA dataset
===============

.. autoclass:: PUMADataSet
   :members:
   :private-members:

"""

import os

#from h5netcdf import File  # h5netcdf somehow writes something into the input file and enlarges it significantly. 
from netCDF4 import Dataset as File
#import pdb
from spectratmo.datasets.base import DataSetGCM

from spectratmo import userconfig

from spectratmo.phys_const import g

from spectratmo.output.global_tmean import GlobalTMeanWithoutBeta

#from spectratmo.output.global_tmean import GlobalTMeanWithBeta

import numpy as np

keys_in_file = {
    'u': 'ua', 'v': 'va', 'w': 'wap', 'z': 'zg', 't': 'ta', 'ps': 'ps'
}


class PUMADataSet(DataSetGCM):
    """Data set for PUMA data.

    Notes
    -----

    Before instantiating this class, at least two files have to be
    prepared:

    - in one of the configuration file of spectratmo (for example
      $HOME/.spectratmo/config.py), the variables `puma_path` and
      `puma_path_result_base` have to be defined.

    - a config.py file for each set of data (for example in
      `puma_path_result_base + '/set0/config.py'`) where an iterable
      `instants` and a str `name` have to be defined.

    parameters
    ----------

    name : str

      Short name of the dataset.

    without_sh : {False, bool}

      Without spherical harmonic operator.

    """

    _classes_output = DataSetGCM._classes_output
    _classes_output[GlobalTMeanWithoutBeta._name] = GlobalTMeanWithoutBeta
    #_classes_output[GlobalTMeanWithBeta._name] = GlobalTMeanWithBeta

    def __init__(self, name='set0',path_file=None, puma_path_result_base=None, without_sh=False,make_eddy=False):


        if path_file==None or puma_path_result_base==None:
            try:
                reload(userconfig)
                self.path_file = os.path.expanduser(userconfig.puma_path)
                puma_path_result_base = os.path.expanduser(
                    userconfig.puma_path_result_base)
            except AttributeError:
                raise userconfig.ConfigError(
                    'The variables `puma_path` and `puma_path_result_base` have to'
                    ' be defined in one of the configuration file.')

        if path_file!=None: self.path_file = path_file
        # if path_file defined in initialisation self.path_file is overwritten


        path = os.path.join(puma_path_result_base, name)

        with File(self.path_file, 'r') as f:
            #nlat = f.dimensions['lat']
            #nlon = f.dimensions['lon']
            #nlev = f.dimensions['lev']
            #self.pressure_levels = f['lev'][:]*100 #hPa to Pa
            #self.lats = f['lat'][:]
            #self.lons = f['lon'][:]

            self.pressure_levels = f.variables['lev'][:]*100 #hPa to Pa
            self.lats = f.variables['lat'][:]
            self.lons = f.variables['lon'][:]
            nlat = self.lats.size
            nlon = self.lons.size
            nlev = self.pressure_levels.size
            # ntime = f.dimensions['ntime']

        super(PUMADataSet, self).__init__(
            name='puma/' + name, path=path,
            nlat=nlat, nlon=nlon, nlev=nlev,
            without_sh=without_sh,planet='Earth',make_eddy=make_eddy)

    def get_spatial3dvar(self, key, instant):


        if key == 'phi':
            key_file = 'zg'
        else:
            try:
                key_file = keys_in_file[key]
            except KeyError:
                raise ValueError('Can not provide key "' + key + '"')

        with File(self.path_file) as f:
            # A variable for one time
            #arr = f[key_file][instant]
            arr = f.variables[key_file][instant]

        if key == 'phi':
            arr = g * arr

        if key == 'wap' or key == 'w':
            arr = 100 * arr #hPa tp Pa

        #print arr.shape

        if self.make_eddy == 1:
           #print "eddy!"
           arr_zon = np.reshape(np.mean(arr,axis=2),(self.nlev,self.nlat,1))
           #arr_edd
           arr = arr-arr_zon
           #sys.exit(1)

        return arr

    def get_spectral3dvar(self, key, instant):
        arr = self.get_spatial3dvar(key, instant)
        return self.oper.sh3d_from_spat3d(arr)

    def get_spatial2dvar(self, key, instant, ip=0):
        if key == 'phi':
            key_file = 'zg'
        else:
            try:
                key_file = keys_in_file[key]
            except KeyError:
                raise ValueError('Can not provide key "' + key + '"')

        with File(self.path_file) as f:
            # A variable for one time
            #arr = f[key_file][instant, ip]
            arr = f.variables[key_file][instant, ip]

        if key == 'phi':
            arr = g * arr

        if key == 'ps':
            arr = 100 * arr #hPa tp Pa
            #print 'ps!!!'
 
        if key == 'wap' or key == 'w':
            arr = 100 * arr #hPa tp Pa

        if self.make_eddy == 1:
           arr_zon = np.reshape(np.mean(arr,axis=1),(self.nlat,1))
           #arr_edd
           arr = arr-arr_zon

        return arr

    def get_spectral2dvar(self, key, instant, ip=0):
        arr = self.get_spatial2dvar(key, instant, ip)
        return self.oper.sh_from_spat(arr)

    def create_array_spat3d(self, value=None):
        """Create an array representing a field in spatial space."""
        if value is None:
            field = np.empty([self.nlev, self.nlat, self.nlon])
        elif value == 'rand':
            field = np.random.randn(self.nlev, self.nlat, self.nlon)
        elif value == 0:
            field = np.zeros([self.nlev, self.nlat, self.nlon])
        else:
            field = value*np.ones([self.nlev, self.nlat, self.nlon])
        return field


if __name__ == '__main__':
    ds = PUMADataSet()

    asp = ds.oper.create_array_spat()
    ash = ds.oper.create_array_sh()
