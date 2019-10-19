
from __future__ import division, print_function

import os
import numpy as np
from time import time
import struct

import pygrib
import netCDF4
import grads
import cPickle as pickle


r_a_real = 6367470. # earth radius (meters)
p_m = 1013. # mean pressure at sea level in hPa
c_p = 1006. # (J/kg/K)
R = 287. # (J/kg/K)
g = 9.80665 # (m/s^2)
Gamma_dah = g/c_p
khi = R/c_p
eps_unit = 7.*10.**(-5)

Omega = 2*np.pi/86164



class Simul:
    """Represent a simulation."""
    def __init__(   self, name_simul='AFES_T639', lmax=15, r_a = r_a_real,
                    nlat=None,nlon=None,
                    nl_order=2,
                    USE_SHT=True
                ):
        self.r_a = r_a # earth radius
        self.p_m = p_m # mean pressure in hPa
        self.c_p = c_p
        self.R = R
        self.g = g
        self.Omega = Omega
        self.Gamma_dah = Gamma_dah
        self.khi = khi
        self.name_simul = name_simul
        self.eps_unit = eps_unit

        HOME = os.environ.get('HOME')
        if HOME=='/home/pierre': # running on pierre-KTH
            self.computer = 'pierre-KTH'
        elif HOME=='/home/mech/augier': # running on a KTH computer
            self.computer = 'KTH'

        if name_simul[0:4]=='AFES':
            # il nous faut un nom de fichier .ctl de la simulation...
            if name_simul[0:9]=='AFES_T639':
                self.bad_files = ['T639.U.day10.2.grd','T639.U.day10.3.grd',
                                  'T639.U.day6.1.grd', 'T639.U.day6.2.grd',
                                  'T639.U.day6.3.grd','T639.U.day7.1.grd',
                                  'T639.U.day9.2.grd']
                if self.computer=='pierre-KTH':
                    self.path_dir = '/Storage/Data_AFES/T639_few_days'
                elif self.computer=='KTH':
                    self.path_dir = '/home/mech/augier/ScratchAugier/Data_AFES/T639_AFES_COMPLET'
                name_file = 'T639.U.day5.1.ctl'
                lmax = 639
                self.lmax = lmax
                # fourth-order horizontal diffusivity
                self.nu4h = 1.14*10**12   # (m^4/s)
                self.delta_t = 1 # (hour)
            elif name_simul[0:10]=='AFES_T1279':
                self.bad_files = []
####                self.path_dir = '/Storage/Data_AFES/T1279_day12'
####                name_file = 'typh.day12.Ured.ctl'
####                self.lmax = 1279
                self.path_dir = '/Storage/Data_AFES/T1279_nlon240nlat120nlev128'
                name_file = 'U.1st.ctl'
                lmax = 79
                self.lmax = lmax
                self.nu4h = 1.14*10**12   # (m^4/s) FAUX!
                self.delta_t = 3 # (hour)

            os.chdir(self.path_dir)

            # start grads
            self.ga = grads.GaNum(Bin='grads',Echo=False,Window=False)

            file_ga = self.ga.open(name_file)
            qh = self.ga.query("ctlinfo")
            self.qh = qh
            self.nt = qh.nt
            self.nz = qh.nz
            self.zlevs = qh.zlevs
            self.plevs = 1000*self.zlevs  # pressure in hPa


            self.nlon = qh.nx
            self.nlat = qh.ny
            u_ga = self.ga.expr('u')
            self.ga('close 1') # close the file
            self.lons=u_ga.grid.lon[0:self.nlon]
            self.lats=u_ga.grid.lat

        elif name_simul[0:5]=='ECMWF':
            if name_simul[0:11]=='ECMWF_T1279':
                self.nu4h = 1.14*10**12   # (m^4/s) ????
                if self.computer=='pierre-KTH':
                    self.path_dir = '/home/pierre/Storage2/Data_ECMWF/T1279'
                elif self.computer=='KTH':
                    self.path_dir = '/home/mech/augier/ScratchAugier/Data_ECMWF/T1279'
                name_file = 'oper_fc_winter.01.grib'
            elif name_simul[0:10]=='ECMWF_T159':
                self.nu4h = 1.14*10**12   # (m^4/s) ????
                if self.computer=='pierre-KTH':
                    self.path_dir = '/home/pierre/Storage2/Data_ECMWF/T159'
                elif self.computer=='KTH':
                    self.path_dir = '/home/mech/augier/ScratchAugier/Data_ECMWF/T159'
                name_file = 'T159.spectral_data.grib'
            else:
                print('bad name_simul, do you mean ECMWF_T1279 or ECMWF_T159 ?')

            grbs = pygrib.open(self.path_dir+'/'+name_file)

            g_shell = grbs[1]
            lmax = g_shell['J']
            self.lmax = lmax
            gpv = g_shell['pv']
            lgpv = len(gpv)
            nz = (lgpv-2)/2
            self.nz = nz
            self.nz_etak = nz

            self.A_km12 = gpv[0:nz+1]
            self.B_km12 = gpv[nz+1:lgpv]

            lAB = len(self.A_km12)

            self.delta_A = self.A_km12[1:lAB] - self.A_km12[0:lAB-1]
            self.delta_B = self.B_km12[1:lAB] - self.B_km12[0:lAB-1]

            self.A_k = (self.A_km12[1:lAB] + self.A_km12[0:lAB-1])/2
            self.B_k = (self.B_km12[1:lAB] + self.B_km12[0:lAB-1])/2

            p_km12approx = self.A_km12 + self.B_km12*101300
            self.plevs_approx = (p_km12approx[0:nz] + p_km12approx[1:nz+1])/2/100

            self.plevs = self.plevs_approx[::-3]
            self.nz_pcoord = len(self.plevs)

#            p_km12approx = self.A_km12 + self.B_km12*103000
#            plevs_approx2 = (p_km12approx[0:nz] + p_km12approx[1:nz+1])/2/100
#            self.plevs = plevs_approx2[::-1]

            grbs.close()

            if USE_SHT:
                # we let shtns compute the correct nlat and nlon
                self.nlat = 0
                self.nlon = 0
            else:
                name_file = name_file[:-5]+'.nc'
                ncf = netCDF4.Dataset(self.path_dir+'/Pressure_coord/'+name_file, 'r')
                nclons = ncf.variables['lons']
                self.nlon = len(nclons)
                self.lons = nclons[:]
                nclats = ncf.variables['lats']
                self.nlat = len(nclats)
                self.lats = nclats[:]

        else:
            self.lmax = lmax
            # !? lmax odd !?

            if nlat==None or nlon==None:
                self.nlat = 0
                self.nlon = 0
            else:
                self.nlat = nlat
                self.nlon = nlon

        if 'plevs' in dir(self):
            self.plevs_bot, self.plevs_top = self.__create_limites_plevs()
            self.delta_p = self.plevs_bot-self.plevs_top
            self.p_M = self.plevs_bot.max()


        if USE_SHT:
            # import module spharm for the spherical harmonics
            import easypyshtns as epshtns
            print('creation of an instance of class easypyshtns.easypysht.')
            t1 = time()
            self.esh = epshtns.easypysht(lmax=self.lmax, 
                                          nlat=self.nlat, nlon=self.nlon, 
                                          nl_order=nl_order)
            t2 = time()
            print('done in {0:3.2f} s'.format(t2-t1))

            esh = self.esh
            self.nlon = esh.nlon
            self.nlat = esh.nlat
            self.lons = esh.lons
            self.lats = esh.lats
            self.order_lat = esh.order_lat
            self.delta = esh.delta

            self.LONS = esh.LONS
            self.LATS = esh.LATS
            self.cosLATS = esh.cosLATS


            self.l_idx = esh.l_idx
            self.l2_idx = esh.l2_idx
            self.m_idx = esh.m_idx
            self.nlm = esh.nlm
            self.idx_lm = esh.idx_lm

            self.init_array_SH = esh.init_array_SH

            self.l2_l = esh.l2_l   
            self.kh_l = esh.kh_l

            self.wavelengths = np.zeros(self.kh_l.shape)
            self.wavelengths[1:] = 2*np.pi/self.kh_l[1:]

            self.print_array_SH = esh.print_array_SH

            self.SH_from_spat = esh.SH_from_spat
            self.spat_from_SH = esh.spat_from_SH
            self.gradf_from_fSH = esh.gradf_from_fSH

            self.spat3D_from_SH3D = esh.spat3D_from_SH3D

            self.uuvv_from_hdivrotSH = esh.uuvv_from_hdivrotSH
            self.uuvv_from_uDuRSH = esh.uuvv_from_uDuRSH

            self.hdivrotSH_from_uuvv = esh.hdivrotSH_from_uuvv
            self.hdivrotSH_from_uDuRSH = esh.hdivrotSH_from_uDuRSH
            self.uDuRSH_from_hdivrotSH = esh.uDuRSH_from_hdivrotSH

            self.spectrum_from_SH = esh.spectrum_from_SH
            self.cospectrum_from_2fieldsSH = esh.cospectrum_from_2fieldsSH
            self.cospectrum_from_2vectorsSH = esh.cospectrum_from_2vectorsSH

            self.chrono_sht = esh.chrono_sht

            self.complex64_save_netCFD = esh.complex64_save_netCFD
            self.convert2npcomplex = esh.convert2npcomplex
            self.convert2complex64_save_netCFD = esh.convert2complex64_save_netCFD

        else:
            self.delta = 360./self.nlon
            self.LONS, self.LATS = np.meshgrid(self.lons, self.lats)

            self.lrange = np.arange(self.lmax+1)
            self.l2_l = self.lrange*(self.lrange+1)
            self.kh_l = np.sqrt(self.l2_l)/self.r_a


        self.cosLATS = np.cos(self.LATS/180*np.pi)





    def __create_limites_plevs(self):
        plevs = self.plevs
        nz = len(plevs)
        plevs_bot = np.ones(nz)
        plevs_bot[0] = plevs[0] + (plevs[0]-plevs[1])/2
        plevs_bot[1:nz-1+1] = (plevs[1:nz-1+1] + plevs[0:nz-2+1])/2
        plevs_top = np.zeros(nz)
        plevs_top[nz-1] = 0.
        plevs_top[0:nz-2+1] = plevs_bot[1:nz-1+1]
        return plevs_bot, plevs_top

    def __create_delta_p(self):
        plevs = self.plevs
        nz = len(plevs)
        delta_p = np.zeros(nz)
        delta_p[0] = p_m - plevs[0] + ( plevs[0]-plevs[1] )/2
        for i in range(1,nz-2 +1):
            delta_p[i] = ( plevs[i-1] - plevs[i+1] )/2
        delta_p[nz-1] = plevs[nz-1]/2 + (plevs[nz-2] - plevs[nz-1])/2
        return delta_p











    def load_grib_file(self, grib_file_name='oper_fc_winter.grib'):
        grbs = pygrib.open(self.path_dir+'/'+grib_file_name)
        return grbs


    def load_beta_ps_mean(self):
        print('load "beta_ps_mean"')
        t1 = time()
        name_save = self.name_simul+'_beta_ps_mean.pickle'
#        os.chdir(self.path_dir)
        f = open(self.path_dir+'/Statistics/'+name_save, 'r')
        dico_save = pickle.load(f)
        f.close()
        self.beta3D = dico_save['beta_ps_mean']
        self.nb_lev_beta = dico_save['nb_lev_beta']

        self.beta_mean1D = np.empty(self.nb_lev_beta)
        for ip in range(self.nb_lev_beta):
            self.beta_mean1D[ip] = self.mean_field(self.beta3D[ip])

        return self.beta3D, self.nb_lev_beta


    def load_meanT_coefAPE(self, season='summer'):
        name_save = self.name_simul+'_meanT_coefAPE.pickle'

        if self.name_simul=='ECMWF_T1279':
            name_save = self.name_simul+'_'+season+'_meanT_coefAPE.pickle'

        f = open(self.path_dir+'/Statistics/'+name_save, 'r')
        dico_save = pickle.load(f)
        f.close()
        self.Tmean = dico_save['Tmean']
        self.TmeanR = dico_save['TmeanR']
        self.ThetaMeanR = dico_save['ThetaMeanR']
        self.Coef_APE_Theta = dico_save['Coef_APE_Theta']
        self.Coef_APE_TT = dico_save['Coef_APE_TT']


    def load_topo(self):
        name_topo_file = 'T'+str(self.lmax)+'.topo.pickle'
        f = open(self.path_dir+'/'+name_topo_file, 'r')
        dico_save = pickle.load(f)
        f.close()
        self.topo = dico_save['topo']
        return self.topo


    def load_var3D(self, name_var='u', day=5, hour=1, season='summer'):

        if self.name_simul[:4]=='AFES':
            if name_var[0] in ['u', 'v', 'T', 'o']:
                name_file, it, ip = self.find_location_shell(name_var, day, hour)
                field3D = self.read_var3D_file(name_file, it)
            elif  name_var[0:3]=='Phi':
                if (not (day==5 or (day==11 and hour<9)) and 
                    self.computer=='pierre-KTH'):
                    raise ValueError, 'problem day hour with geopotential'
                if hour>=0 and hour<9:
                    num_file = 1
                elif hour>=9 and hour<17:
                    num_file = 2
                elif hour>=17 and hour<25:
                    num_file = 3
                name_file = (self.name_simul+'_Phi_day'+str(day)+'.'+
                             str(num_file)+'.nc')
                path_file = self.path_dir+'/Complete_data/Phi/'+name_file
                if not os.path.exists(path_file):
                    raise ValueError('file '+path_file+' not found.')
                ncf = netCDF4.Dataset(path_file, 'r')
                ncPhi3D = ncf.variables['Phi3D']
                ih = hour-1 -(num_file-1)*8
                field3D = np.array(ncPhi3D[ih], dtype=np.float64)
                ncf.close()

            for ip in range(self.nz):
                    if abs(field3D[ip]).max()==0. :
                        print('!!!!!!!!\n!!!!!!!!    PROBLEM !!!!,'
                              'field3D[{0}] is full of zeros !'.format(ip))

        elif self.name_simul[:5]=='ECMWF':
            hour = 12
            if not name_var in ['u', 'v', 'T', 'o', 'Phi']:
                raise ValueError, 'name_var should be in [''u'', ''v'', ''T'', ''o'', ''Phi'']'
            else:
                day = repr(int(day))

                if season[0]=='s': season = 'summer'
                else: season = 'winter'

                if self.name_simul=='ECMWF_T159':
                    name_file = 'T159.spectral_data.nc'
                elif self.name_simul=='ECMWF_T1279':
                    name_file = 'oper_fc_'+season+'.'+day.zfill(2)+'.nc'
                    if int(day)==0:
                        day = '1'
                        name_file = 'oper_an_'+season+'.'+day.zfill(2)+'.nc'


                ncf = netCDF4.Dataset(self.path_dir+'/Pressure_coord/'+name_file, 'r')
                if name_var[0]=='u':
                    ncvar_lm3D  = ncf.variables['uu_lm3D']
                elif name_var[0]=='v':
                    ncvar_lm3D  = ncf.variables['vv_lm3D']
                elif name_var[0]=='o':
                    ncvar_lm3D  = ncf.variables['oo_lm3D']
                elif name_var[0]=='T':
                    ncvar_lm3D  = ncf.variables['TT_lm3D']
                elif name_var=='Phi':
                    ncvar_lm3D  = ncf.variables['Phi_lm3D']

                field_lm3D = self.convert2npcomplex(ncvar_lm3D[:])
                field3D = self.spat3D_from_SH3D(field_lm3D)

        for_output1 = 'load var3D type \"'+name_var+'\",'
        print(for_output1.ljust(25) +'day = '+repr(day)+
              ', hour = '+repr(hour)+' (file '+name_file+')')
        return field3D



    def load_shell(self, name_var='u', day=5, hour=1, num_level=1, 
                   grbs=None, season='summer'):
        """Load a shell from a file."""
        if self.name_simul[0:4]=='AFES' and name_var[0] in ['u', 'v', 'T', 'o']:
            name_file, it, ip = self.find_location_shell(
                name_var, day, hour, num_level)
            field = self.read_shell_file(name_file, it, ip)
            obj_shell = 1
            for_output = (', plev = {0:7.1f} hPa'.format(self.plevs[ip])+
                          ' (file '+name_file+')')
            for_output1 = 'load shell type \"'+name_var[0]+'\",'
            print(for_output1.ljust(25) +'day = '+repr(day)+
                  ', hour = '+repr(hour)+for_output)


        elif self.name_simul[0:4]=='AFES' and name_var=='ps':
            if self.name_simul[0:9]=='AFES_T639':
                # the days with correct data are 1,2,3,4,5,6 and 11
                # for 11 only the 10 first hours
                if day<6:
                    name_file = 'T639.out01.PS.grd'
                elif day<11:
                    name_file = 'T639.out02.PS.grd'
                else:
                    name_file = 'T639.out03.PS.grd'
                if not os.path.exists(name_file):
                    raise ValueError('.grd file \"{0}/{1}\" not found'.format(
                        self.path_dir, name_file))
                iday_file = (day-1)%5
                it = 24*iday_file + hour -1
            elif self.name_simul[0:10]=='AFES_T1279':
                name_file = 'PS.grd'
                if hour/3.%1>0.:
                    raise ValueError(
                        'hour = {0} is not present in the data'.format(hour))
                if day>34:
                    raise ValueError(
                        'day = {0} superior that day_max = 17'.format(day))
                it = (day-1)*8 + hour/3 -1
                if it>271:
                    raise ValueError(
                        'it_file = {0} superior that it_file = 136'.format(day))

            field = self.read_ps_file(name_file, it)
            field = field/100. # (hPa)
            obj_shell = 1
            for_output1 = 'load shell type \"'+name_var[0:2]+'\",'
            print(for_output1.ljust(25) +'day = '+repr(day)+
                  ', hour = '+repr(hour)+' (file '+name_file+')')

        elif self.name_simul[0:4]=='AFES' and not name_var=='Phi':
            if name_var[-6:]=='direct':
                name_var = name_var[0]
            obj_shell = ShellAtm(self, name_var, day=day, hour=hour, 
                                 num_level=num_level)
            field = np.array(obj_shell.data)
            if name_var in ['hcurl', 'hdiv']:
                field[0, :] = 0.
                field[:, 0] = 0.
                field[-1, :] = 0.
                field[:, -1] = 0.
            if name_var=='ps':
                field = field/100. # (hPa)

        elif self.name_simul[0:4]=='AFES' and name_var=='Phi':
            if not (day==5 or (day==11 and hour<9)):
                raise ValueError, 'problem day hour with geopotential'
            if hour>=0 and hour<9:
                num_file = 1
            elif hour>=9 and hour<17:
                num_file = 2
            elif hour>=17 and hour<25:
                num_file = 3
            name_file = (self.name_simul+'_Phi_day'+str(day)+'.'+
                         str(num_file)+'.nc')
            path_file = self.path_dir+'/Complete_data/'+name_file
            if not os.path.exists(path_file):
                raise ValueError('file '+path_file+' not found')

            ncf = netCDF4.Dataset(path_file, 'r')
            obj_shell = 1

            ncPhi3D = ncf.variables['Phi3D']

            ih = hour-1 -(num_file-1)*8
            field = ncPhi3D[ih,num_level-1]
            ncf.close()

            for_output = ' (file '+name_file+')'
            if 'plev' in dir(self):
                for_output = ', plev = {0:7.1f} hPa'.format(self.plev)+for_output

            for_output1 = 'load shell type \"'+name_var+'\",'

            print(for_output1.ljust(25) +'day = '+repr(day)+
                  ', hour = '+repr(hour)+for_output)

        elif self.name_simul[0:5]=='ECMWF':
            if season[0]=='s': 
                season = 'summer'
                self.month = 8
            else: 
                season = 'winter'
                self.month = 12

            hour = 12
            if  grbs==None and name_var in ['u', 'v', 'T', 'o', 
                                            'Phi', 'ps', 'Phis']:
                day = repr(int(day))

                if self.name_simul=='ECMWF_T159':
                    name_file = 'T159.spectral_data.nc'
                elif self.name_simul=='ECMWF_T1279':
                    name_file = 'oper_fc_'+season+'.'+day.zfill(2)+'.nc'
                    if int(day)==0:
                        day = '1'
                        name_file = 'oper_an_'+season+'.'+day.zfill(2)+'.nc'

                obj_shell = name_file

                ncf = netCDF4.Dataset(self.path_dir+'/Pressure_coord/'+name_file, 'r')
                if name_var[0]=='u':
                    ncvar_lm2D  = ncf.variables['uu_lm3D'][num_level-1]
                elif name_var[0]=='v':
                    ncvar_lm2D  = ncf.variables['vv_lm3D'][num_level-1]
                elif name_var[0]=='o':
                    ncvar_lm2D  = ncf.variables['oo_lm3D'][num_level-1]
                elif name_var[0]=='T':
                    ncvar_lm2D  = ncf.variables['TT_lm3D'][num_level-1]
                elif name_var=='Phi':
                    ncvar_lm2D  = ncf.variables['Phi_lm3D'][num_level-1]
                elif name_var=='ps':
                    ncvar_lm2D  = ncf.variables['ps_lm']
                elif name_var=='Phis':
                    ncvar_lm2D  = ncf.variables['Phis_lm']

                field_lm = self.convert2npcomplex(ncvar_lm2D[:])
                field = self.spat_from_SH(field_lm)


                for_output = ' (file '+name_file+')'
                if 'plev' in dir(self):
                    for_output = ', plev = {0:7.1f} hPa'.format(self.plev)+for_output

                for_output1 = 'load shell type \"'+name_var+'\",'

                print(for_output1.ljust(25) +'day = '+repr(day)+for_output)


            elif grbs==None:
                raise ValueError(
                    'with ECMWF, argument grds has to be given\n'
                    '    >>> grds= sim.load_grib_file(\'file.grib\')\n'
                    '    >>> ps, grb = sim.load_shell('
                    'name_var=\'ps\', num_level=1)')

            else:
                if num_level>self.nz:
                    raise ValueError, 'num_level>self.nz'

                if name_var=='ps':
                    ig_file = 1
                    if grbs[1]['name']=='Geopotential':
                        ig_file += 1
                elif name_var in ['hrot', 'hdiv', 'T']:
                    if name_var=='hrot':
                        coef_name_var = 0
                    elif name_var=='hdiv':
                        coef_name_var = 1
                    elif name_var=='T':
                        coef_name_var = 2
                    ig_file = int(2 + 3*(num_level-1) + coef_name_var)
                    if grbs[1]['name']=='Geopotential':
                        ig_file += 1
                else:
                    raise ValueError('not a proper name_var')

                print('ig_file =', ig_file)

                grb = grbs[ig_file]
                obj_shell = grb
                print('ig_file='+repr(ig_file)+' level='+repr(grb['level'])+
                      ' ('+grb['typeOfLevel']+') '+grb['name'])

                field_lmr = +grb.values[0::2]
                field_lmi = +grb.values[1::2]

                field_lm = self.init_array_SH()
                field_lm.real = field_lmr
                field_lm.imag = field_lmi

                # rotation around North-South axe
                field_lm = field_lm*np.exp(1j * self.m_idx*np.pi)

                # Warning: with ECMWF, load_shell() returns a field_lm
                # (in spectral space) (only true for hdiv, hrot and T,
                # and if a grds object is given...)
                field = field_lm




        if field.min()==0. :
            print('!!!!!!!!\n!!!!!!!!    PROBLEM !!!!,'
                  'field is full of zeros !\n!!!!!!!!')
        return field, obj_shell





    def find_location_shell(self, name_var, day, hour, num_level=1):
        ip = num_level-1
        if self.name_simul[0:9]=='AFES_T639':
            if name_var[0] in ['u', 'v', 'T', 'o']:
                num_file = int((hour-1)/8.)+1
                name_file = ('T'+repr(self.lmax)+'.'+
                             name_var[0].capitalize()+'.day'+repr(day)+'.'+
                             repr(num_file)+'.grd')
                it = (hour-1)%8
            else:
                print('problem...')

        elif self.name_simul[0:10]=='AFES_T1279':
            if name_var[0] in ['u', 'v', 'T', 'o']:
                name_file = name_var[0].capitalize()+'.1st.grd'
                if name_var=='o':
                    name_file = 'OMG.1st.ctl'
                if hour/3.%1>0.:
                    raise ValueError(
                        'hour = {0} is not present in the data'.format(hour))
                if day>17:
                    raise ValueError(
                        'day = {0} superior that day_max = 17'.format(day))
                it = (day-1)*8 + hour/3 -1
                if it>135:
                    raise ValueError(
                        'it = {0} superior that it = 135'.format(day))
        if name_file in self.bad_files:
            print('\nWarning: possibly file with many zeros...\n')
        path_file = self.path_dir+'/'+name_file
        if not os.path.exists(path_file):
            raise ValueError('.grd file '+path_file+' not found')
        return name_file, it, ip

    def read_shell_file(self, name_file, it, ip):
        path_file = self.path_dir+'/'+name_file
        size_file = os.path.getsize(path_file)
        nlon = self.nlon
        nlat = self.nlat
        shape = (nlat, nlon)
        nb_values_to_read = nlon*nlat
        nz = self.nz
        if self.name_simul=='AFES_T639':
            nt = 8
        elif self.name_simul=='AFES_T1279':
            nt = 136
        size = 4  # float
        endian = '>' # code for big-endian
        nb_bytes_to_read = size*nb_values_to_read
        if not size_file==nlon*nlat*nz*nt*size:
            raise ValueError, 'Problem: check file structure?'
        ibyte_start = (it*nz+ip)*nb_bytes_to_read
        with open(path_file,mode='rb') as f:
            f.seek(ibyte_start)
            content = f.read(nb_bytes_to_read)
            values = struct.unpack(endian+repr(nb_values_to_read)+'f', 
                                   content[:nb_bytes_to_read])
            f.close()
        field2D = np.array(values).reshape(shape)
        return field2D




    def read_ps_file(self, name_file, it):
        path_file = self.path_dir+'/'+name_file
        size_file = os.path.getsize(path_file)
        nlon = self.nlon
        nlat = self.nlat
        shape = (nlat, nlon)
        nb_values_to_read = nlon*nlat
        if self.name_simul=='AFES_T639':
            nt = 120
        elif self.name_simul=='AFES_T1279':
            nt = 272
        size = 4  # float
        endian = '>' # code for big-endian
        nb_bytes_to_read = size*nb_values_to_read
        if not size_file==nlon*nlat*nt*size:
            print(name_file, '\nsize_file', size_file,
                  nlon*nlat*nt*size)

            raise ValueError('Problem: check file structure?')
        ibyte_start = it*nb_bytes_to_read
        with open(path_file,mode='rb') as f:
            f.seek(ibyte_start)
            content = f.read(nb_bytes_to_read)
            values = struct.unpack(endian+repr(nb_values_to_read)+'f', 
                                   content[:nb_bytes_to_read])
            f.close()
        field2D = np.array(values).reshape(shape)
        return field2D



    def read_var3D_file(self, name_file, it):
        path_file = self.path_dir+'/'+name_file
        size_file = os.path.getsize(path_file)
        nlon = self.nlon
        nlat = self.nlat
        nz = self.nz
        shape = (nz, nlat, nlon)
        nb_values_to_read = nz*nlon*nlat
        if self.name_simul=='AFES_T639':
            nt = 8
        elif self.name_simul=='AFES_T1279':
            nt = 136
        size = 4  # float
        endian = '>' # code for big-endian
        nb_bytes_to_read = size*nb_values_to_read
        if not size_file==nlon*nlat*nz*nt*size:
            raise ValueError, 'Problem: check file structure?'
        ibyte_start = it*nb_bytes_to_read
        with open(path_file,mode='rb') as f:
            f.seek(ibyte_start)
            content = f.read(nb_bytes_to_read)
            values = struct.unpack(endian+repr(nb_values_to_read)+'f', 
                                   content[:nb_bytes_to_read])
            f.close()
        field3D = np.array(values).reshape(shape)
        return field3D









    def mean_field(self, field):
        delta = 360./self.nlon
        return (delta*np.pi/180)**2 * np.sum(field*self.cosLATS) /(4*np.pi)

    def mean_field_representative(self, field, ip=None, beta=None):
        if beta==None:
            if not 'beta3D' in dir(self):
                raise ValueError('first run beta3D, nb_lev_beta = sim.load_beta_ps_mean()')
            if ip<self.nb_lev_beta:
                result = self.mean_field(field*self.beta3D[ip])/self.beta_mean1D[ip]
            elif ip>=self.nb_lev_beta:
                result = self.mean_field(field)
        else:
            result = self.mean_field(field*beta)/self.mean_field(beta)
        return result


    def vertical_derivative_f(self, f):
#        if not f.shape==self.plevs.shape:
#            raise ValueError, 'we hate to have f.shape==plevs.shape'
        plevs = self.plevs
        nz = len(plevs)
        df_dp = np.zeros(f.shape)
        for ip in np.arange(1,nz-1):
            df_dp[ip] = ( ( f[ip-1] - f[ip+1] ) / 
                             ( plevs[ip-1]    - plevs[ip+1] ) )
#        df_dp[1:nz-1] = ( ( f[0:nz-2] - f[2:nz] ) / 
#                             ( plevs[0:nz-2]    - plevs[2:nz] ) )
        df_dp[0] = ( f[0] - f[1] ) / ( plevs[0] - plevs[1] ) 
        df_dp_int = ( f[nz-2] - f[nz-1] ) / ( plevs[nz-2] - plevs[nz-1] ) 
        p_int = (plevs[-1]+plevs[-2])/2
        d2f_dp2 = (df_dp[-2]-df_dp_int)/(plevs[-2]-p_int)
        df_dp[-1] = df_dp_int + d2f_dp2*(plevs[-1]-p_int)
        return df_dp


    def init_array_spat(self, value = None):
        """Initialise the spatial array."""
        if value==None:
            field = np.empty([self.nlat, self.nlon])
        elif value==0:
            field = np.zeros([self.nlat, self.nlon])
        else:
            field = value*np.ones([self.nlat, self.nlon])
        return field





    def print_info(self):
        print('\nInformation on the simulation : \"'+self.name_simul+'\"'+
              'nlon = {0}, nlat = {1}'.format(self.nlon, self.nlat)+
              '1 point every {0:6.2f} km'.format(
                  2*np.pi*r_a_real/self.nlon/1000))
        if hasattr(self, 'nz'):
            print('{0} levels in the vertical'.format(self.nz))
        print('')
        if hasattr(self, 'esh'):
            print('Info library shtns (spherical harmonic transforms):')
            self.esh.print_info()
            print('')


















    def complete_fields_spat_SH_deSH(self, fields, fields_lm=None, 
                                     arrays_deSH=None):
        """complete the SetOfVariables instances fields, fields_lm and
        arrays_deSH as much as possible from field to field_lm

        """
        if fields_lm==None:
            fields_lm_WAS_GIVEN = False
            fields_lm = SetOfVariables('coef spherical harmonics')
        else:
            fields_lm_WAS_GIVEN = True
        for key, item_data in fields.ddata.iteritems():
            if not fields_lm.ddata.has_key(key+'_lm'):
                fields_lm.ddata[key+'_lm'] = self.SH_from_spat(item_data)



        for s in fields.ddata.keys(): 
            if s[:2]=='uu':
                s_complement = s[2:]
                if (    not fields_lm.ddata.has_key('hrot'+s_complement+'_lm') or 
                        not fields_lm.ddata.has_key('hdiv'+s_complement+'_lm')):
                    hdiv_lm, hrot_lm = self.hdivrotSH_from_uuvv(
                                        fields.ddata['uu'+s_complement], 
                                        fields.ddata['vv'+s_complement])
                    fields_lm.ddata['hdiv'+s_complement+'_lm'] = hdiv_lm
                    fields_lm.ddata['hrot'+s_complement+'_lm'] = hrot_lm

        # from field_lm to  array_deSH
        arrays_deSH = SetOfVariables('energy density')
        for key, item_data in fields_lm.ddata.iteritems():
            if not arrays_deSH.ddata.has_key(key+'_de'):
                arrays_deSH.ddata[key+'_de'] = \
                    self.esh._array_deSH_from_SH(item_data, key)

        if fields_lm_WAS_GIVEN:
            return arrays_deSH
        else:
            return fields_lm, arrays_deSH


    def spectra_from_deSH(self, arrays_deSH):
        En = SetOfVariables('spectra')
        En.lrange = self.lrange
        for key, item_data in arrays_deSH.ddata.iteritems():
            En.ddata['E_'+key[:-6]+'_l'] = np.zeros(self.lmax+1)
            En.ddata['E_'+key[:-6]+'_l'] = self.esh._spectrum_from_array_deSH(item_data)
#        En.ddata['E_K_l'] = En.ddata['E_uu_l'] + En.ddata['E_vv_l']
        return En























class ShellAtm:
    """Represent a shell."""
    def __init__(self, sim, name_var='u', day=6, hour=1, num_level=1):

        # some checks...
        liste_name_var = ['u', 'v', 'T', 'hcurl', 'hdiv', 'ps', 'o']
        if not name_var in liste_name_var:
            text_error = '\"{0}\" is not a proper name of ga variable'.format(name_var)
            raise ValueError, text_error

        if sim.name_simul[0:9]=='AFES_T639':
            if not (day==5 or day==11 or day==15) and not name_var=='ps':
                raise ValueError, 'problem with the day number (?), check files...'
            if hour<1 or hour>24:
                raise ValueError, 'bad hour value, hour = '+repr(hour)
            if (day==11 and hour>8):
                raise ValueError, 'bad hour value with day=11, hour = '+repr(hour)
            if num_level<1 or num_level>sim.nz:
                raise ValueError, 'bad level value, num_level = '+repr(num_level)
        if sim.name_simul[0:10]=='AFES_T1279':
            print('no check for AFES_T1279... '
                  'that could be nice to implement them...')

        # go in the directory and find the name of the file
        os.chdir(sim.path_dir)
        name_file, name_file2, it_file = self.find_name_file(
            sim, name_var, day, hour)

        file_ga = sim.ga.open(name_file)              # open file
        if name_var in ['hcurl', 'hdiv']:
            file_ga2 = sim.ga.open(name_file2)

        if name_var in ['u', 'v', 'T', 'o', 'ps']:
            ga_exp = name_var
        elif name_var=='hcurl':
            ga_exp = 'hcurl(u.1, v.2)'
        elif name_var=='hdiv':
            ga_exp = 'hdivg(u.1, v.2)'


        self.nz = file_ga.nz
        if (num_level<1 or self.nz<num_level) and not name_var=='ps':
            raise ValueError,'num_level<1 or nz<num_level, problem...'

        self.nt_file = file_ga.nt
        if it_file<1 or self.nt_file<it_file:
            print('nt_file =', self.nt_file, ' ; it_file =', it_file)
            raise ValueError('it_file<1 or it_file<nt_file, problem...')

        # choice of the time in the file
        sim.ga('set t '+str(it_file))
        # choice of the pressure level
        if name_var=='ps':
            self.ilev = 1
        else:
            zlevs = sim.qh.zlevs
            zlev = zlevs[num_level-1]
            self.ilev = num_level-1
            sim.ga('set lev '+str(zlev))

        self.obj_ga = sim.ga.expr(ga_exp)

        if not name_var=='ps':
            self.zlev = self.obj_ga.grid.lev[0]
            self.plev = 1000*self.zlev

        if name_var in ['hcurl', 'hdiv']:
            sim.ga('close 2')
        sim.ga('close 1') # close the file

        self.data = self.obj_ga.data
        # variables are cyclic, we don't want that...
        self.data = self.data[:, 0:sim.nlon]
        self.name_file = name_file

        self.day = day
        self.hour = hour

        for_output = ' (file '+name_file+')'
        if not name_file2=='':
            for_output = ' (files '+name_file+' and '+name_file2+')'
        if 'plev' in dir(self):
            for_output = ', plev = {0:7.1f} hPa'.format(self.plev)+for_output

        for_output1 = 'load shell type \"'+name_var+'\",'

        print(for_output1.ljust(25) +'day = '+repr(day)+
              ', hour = '+repr(hour)+for_output)





    def find_name_file(self, sim, name_var='u', day=6, hour=1):

        name_file2 = ''
        if sim.name_simul[0:9]=='AFES_T639':
            if name_var in ['u', 'v', 'T', 'o']:
                num_file = int((hour-1)/8.)+1
                name_file = ('T'+repr(sim.lmax)+'.'+
                             name_var.capitalize()+'.day'+repr(day)+'.'+
                             repr(num_file)+'.ctl')
                it_file = (hour-1)%8 + 1

            elif name_var in ['hcurl', 'hdiv']:
                name_file = 'T'+repr(sim.lmax)+'.'+\
                        'U.day'+repr(day)+'.1.ctl'
                name_file2 = 'T'+repr(sim.lmax)+'.'+\
                        'V.day'+repr(day)+'.1.ctl'

            elif name_var=='ps':
                if day<6:
                    name_file = 'T639.out01.PS.ctl'
                elif day<11:
                    name_file = 'T639.out02.PS.ctl'
                else:
                    name_file = 'T639.out03.PS.ctl'
                if not os.path.exists(name_file):
                    raise ValueError('.ctl file \"{0}/{1}\" not found'.format(
                        self.path_dir, name_file))
                iday_file = (day-1)%5
                it_file = 24*iday_file + hour


        elif sim.name_simul[0:10]=='AFES_T1279':
            if name_var in ['u', 'v', 'T', 'o']:
                name_file = name_var.capitalize()+'.1st.ctl'
                if name_var=='o':
                    name_file = 'OMG.1st.ctl'
                if hour/3.%1>0.:
                    raise ValueError(
                        'hour = {0} is not present in the data'.format(hour))
                if day>17:
                    raise ValueError(
                        'day = {0} superior that day_max = 17'.format(day))
                it_file = (day-1)*8 + hour/3
                if it_file>136:
                    raise ValueError(
                        'it_file = {0} superior that it_file = 136'.format(day))

            elif name_var in ['hcurl', 'hdiv']:
                name_file  = 'U.1st.ctl'
                name_file2 = 'V.1st.ctl'

            elif name_var=='ps':
                name_file = 'PS.ctl'
                if hour/3.%1>0.:
                    raise ValueError(
                        'hour = {0} is not present in the data'.format(hour))
                if day>34:
                    raise ValueError(
                        'day = {0} superior that day_max = 17'.format(day))

                it_file = (day-1)*8 + hour/3
                if it_file>272:
                    raise ValueError(
                        'it_file = {0} superior that it_file = 136'.format(day))

        if name_file2=='':
            if not os.path.exist(name_file):
                raise ValueError(
                    '.ctl file \"{0}/{0}\" not found'.format(
                        sim.path_dir, name_file))
        else:
            if not os.path.exist(name_file) or not os.path.exist(name_file2):
                raise ValueError('at least one of the 2 files not found')


        return name_file, name_file2, it_file
















class SetOfVariables:
    """ A simple class for a set of variables """
    def __init__(self, name_type_variables):
        self.name_type_variables = name_type_variables
        self.ddata = dict()

    def __add__(self, other):
        if isinstance(other, SetOfVariables): 
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]+other.ddata[k]
        elif isinstance(other, int) or isinstance(other, float): 
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]+other

        else: # just to avoid a bug with SetOfVariables created with previous version...
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]+other.ddata[k]

        return obj_result
    __radd__ = __add__
    def __sub__(self, other):
        if isinstance(other, SetOfVariables): 
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]-other.ddata[k]
        return obj_result  

    def __mul__(self, other):
        if isinstance(other, (int, float)): 
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = other*self.ddata[k]
        return obj_result  
    __rmul__ = __mul__ 
    def __div__(self, other):
        if isinstance(other, (int, float)): 
            obj_result = SetOfVariables(self.name_type_variables)
            for k, item_data in self.ddata.iteritems():
                obj_result.ddata[k] = self.ddata[k]/other
        return obj_result  

    def add_APE_spectrum(self, Tmean, GammaMean):
        self.ddata['E_APE_n'] = ( self.ddata['E_TT_n']
                                 *g/(Tmean*(Gamma_dah-GammaMean))
                                )
        self.ddata['E_APEb_n'] = ( self.ddata['E_TTb_n']
                                 *g/(Tmean*(Gamma_dah-GammaMean))
                                )

    def reset_to_zeros(self):
        for key, item_data in self.ddata.iteritems():
            if np.isscalar(item_data):
                del(self.ddata[key])
                self.ddata[key] = 0.
            else:
                item_data[:] = 0.













