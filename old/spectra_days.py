#!/usr/bin/env python
# to use for example with ipython -pylab
# run /home/pierre/Python/Production/Spectra/spectra_days.py

# import basic modules...
import os, sys
import numpy as np
# import homemade module to load data and perform spharm transform
import treat_simul_shtns22 as treat_simul
# import function time in module time for timming functions
from time import time
# import module for saving...
####import pickle
import cPickle as pickle

import matplotlib.pyplot as plt



SAVE = 1

season = 'summer'
#season = 'winter'


#name_simul = 'AFES_T639'
#name_simul = 'AFES_T1279'
name_simul = 'ECMWF_T1279'
#name_simul = 'ECMWF_T159'
Osim = treat_simul.treat_simul(name_simul=name_simul)



plevs = Osim.plevs
nz = len(plevs)
p00 = 1000   # hPa
A = (p00/plevs)**Osim.khi

beta3D_small, nb_lev_beta = Osim.load_beta_ps_mean()
beta3D = np.ones([nz, Osim.nlat, Osim.nlon], dtype=np.float32)
beta3D[0:nb_lev_beta] = beta3D_small


Osim.load_meanT_coefAPE(season=season)
Coef_APE_Theta = Osim.Coef_APE_Theta
Coef_APE_TT = Osim.Coef_APE_TT




def zeros_list_Ens_spectra():
    list_Ens_spectra = []
    for ip in range(nz):
        Ens_spectra = treat_simul.ensemble_variables('spectra')
    #    Ens_spectra.ddata['E_l'] = np.zeros([Osim.lmax+1])
    #    Ens_spectra.ddata['E_K_l'] = np.zeros([Osim.lmax+1])
    #    Ens_spectra.ddata['E_A_l'] = np.zeros([Osim.lmax+1])
        Ens_spectra.ddata['E_uu_l'] = np.zeros([Osim.lmax+1])
        Ens_spectra.ddata['E_vv_l'] = np.zeros([Osim.lmax+1])
        Ens_spectra.ddata['E_TT_l'] = np.zeros([Osim.lmax+1])
        Ens_spectra.ddata['E_hrot_l'] = np.zeros([Osim.lmax+1])
        Ens_spectra.ddata['E_hdiv_l'] = np.zeros([Osim.lmax+1])
        list_Ens_spectra.append(Ens_spectra)
    return list_Ens_spectra






if name_simul[:4]=='AFES':
    if Osim.computer=='pierre-KTH':
        days = [5,11,15]
    elif Osim.computer=='KTH':
        days = [3,4,5,8,11,12,13,14,15]
elif name_simul=='ECMWF_T1279':
    days = [1, 5, 10, 15, 20, 25]
    days = [0]

elif name_simul=='ECMWF_T159':
    days = [5]

#days = [5]





def hour_max(day):
    if day==11:
        h = 8
    else:
        h = 24
    return h

delta_hour = 8
def hours_day_namesimul(day, name_simul):
    if name_simul[:4]=='AFES':
        hours = np.arange(1, hour_max(day)+1, delta_hour)
    elif name_simul[:5]=='ECMWF':
        hours = [12]
    return hours




nbtot = 0
for day in days:
    hours = hours_day_namesimul(day, name_simul)
    nbtot += len(hours)

nb_to_do = nbtot+0 # "deepcopy"...
nb_instants_computed = 0
for day in days:
    list_Ens_spectra = zeros_list_Ens_spectra()
    hours = hours_day_namesimul(day, name_simul)
    nb_instants_computed_day = 0
    for hour in hours:

        print 'calcul spectra for day = {0:2n}, hour = {1:2n}'.format(day,hour)
        t1_1time = time()
        uu3D = Osim.load_var3D(name_var='u', day=day, hour=hour, season=season)
        vv3D = Osim.load_var3D(name_var='v', day=day, hour=hour, season=season)
        TT3D = Osim.load_var3D(name_var='T', day=day, hour=hour, season=season)

        uu3D = uu3D*beta3D
        vv3D = vv3D*beta3D
        for ip in range(nz):
            TT3D[ip] = (TT3D[ip]-Osim.mean_field_representative(TT3D[ip], ip))*beta3D[ip]

        for ip in range(nz):
            fields = treat_simul.ensemble_variables('fields in spatial space')

            fields.ddata['uu'] = uu3D[ip]
            fields.ddata['vv'] = vv3D[ip]
            fields.ddata['TT'] = TT3D[ip]

            print '\r    calcul spectra for p = {0:4.0f} hPa'.format(plevs[ip]),
            sys.stdout.flush()
            fields_lm, arrays_deSH = Osim.complete_fields_spat_SH_deSH(fields)
            Ens_spectra = Osim.spectra_from_deSH(arrays_deSH)
            list_Ens_spectra[ip] += Ens_spectra

        nb_instants_computed_day += 1
        nb_instants_computed += 1
        nb_to_do = nb_to_do - 1


        t2_1time = time()
        print '\n1 time treated in {0:3.2f} s'.format(t2_1time-t1_1time)

        if nb_instants_computed%(nbtot/100.)<1.:
            print 'instant day =', day, 'hour =', hour, ' completed: {0:3.0f}% done'.format(nb_instants_computed/float(nbtot)*100.)

        if not nb_to_do==0:
            print 'there are still {0} instants to treat'.format(nb_to_do)
            print 'approximative time left: {0:5.0f} s'.format((t2_1time-t1_1time)*nb_to_do)
        else:
            print 'computation day = {0} completed...'.format(day)

        del(uu3D)
        del(vv3D)
        del(TT3D)


    for ip in range(nz):
        Ens_spectra = list_Ens_spectra[ip]/nb_instants_computed_day
        Ens_spectra.ddata['E_K_l'] = Ens_spectra.ddata['E_uu_l'] + Ens_spectra.ddata['E_vv_l']
        Ens_spectra.ddata['E_A_l'] = Ens_spectra.ddata['E_TT_l'] * Coef_APE_TT[ip]
        Ens_spectra.ddata['E_l'] = Ens_spectra.ddata['E_K_l'] + Ens_spectra.ddata['E_A_l']
        list_Ens_spectra[ip] = Ens_spectra



    if SAVE:
        name_directory_save = Osim.path_dir+'/Statistics/Spectra'
        if name_simul=='ECMWF_T1279':
            name_directory_save = Osim.path_dir+'/Statistics/Spectra_'+season
        if not os.path.exists(name_directory_save):
            os.mkdir(name_directory_save)
        name_save = 'T'+str(Osim.lmax)+'_spectra_day'+str(day)+'.pickle'
        dico_save = dict([  ['list_Ens_spectra', list_Ens_spectra],
                            ['name_save', name_save]
                        ])
        f = open(name_directory_save+'/'+name_save, 'w')
        pickle.dump(dico_save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    print '\nday = '+str(day)+' is done (nb_instants_computed_day =', nb_instants_computed_day, '\b)\n'







