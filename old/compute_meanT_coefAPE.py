#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# to use for example with ipython -pylab
# run /home/pierre/Python/Production/compute_beta.py

import numpy as np

import os, subprocess

import treat_simul_shtns22 as treat_simul

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#import pickle
import cPickle as pickle
from time import time

import copy

# Heaviside function
H=lambda x : (np.sign(x)+1.)/2 
# smooth Heaviside function
H_smooth = lambda x, delta : (1.+ np.tanh(2*np.pi*x/delta) )/2 


#name_simul = 'AFES_T639'
#name_simul = 'AFES_T1279'
name_simul = 'ECMWF_T1279'
name_simul = 'ECMWF_T159'
Osim = treat_simul.treat_simul(name_simul=name_simul)

COMPUTE_BETA = 'raw'
COMPUTE_BETA = 'tilde'
COMPUTE_BETA = 'phys_smooth'
#COMPUTE_BETA = 'ps_temp_mean'
COMPUTE_BETA = 'beta_ps_mean'


if Osim.name_simul=='ECMWF_T1279':
    days = [1, 5, 10, 15, 20, 25]
    hours = [12]
elif Osim.name_simul=='ECMWF_T159':
    days = [1]
    hours = [12]
elif Osim.name_simul=='AFES_T639':
    day_start = 1
    day_end = 10
    hour_max = 24
    days = range(day_start,day_end+1)
    hours = range(1,hour_max+1)

plevs = Osim.plevs
# we treat only the levels such as plevs > min(ps)
# we take min(ps)\simeq 500 hPa
nb_lev_beta = plt.mlab.find(plevs>500).shape[0]



if COMPUTE_BETA=='raw':

    #beta = Osim.init_array_spat()
    beta = np.zeros([Osim.nlat, Osim.nlon, nb_lev_beta])

    nb_realisation = np.zeros(nb_lev_beta, int)

    for day in days:
        for hour in hours:
            ps, Oshell = Osim.load_shell(name_var='ps', day=day, hour=hour)
            if ps.min()==0. :
                print 'problem, ps is full of zeros !'
            else:
                for ip in range(nb_lev_beta):
                    beta[:,:, ip] += H(ps-plevs[ip])
                    nb_realisation[ip] += 1


    for ip in range(nb_lev_beta):
        beta[:,:,ip] = beta[:,:,ip]/nb_realisation[ip]


    print 'save "beta_raw"'
    t1 = time()
    name_save = 'AFES_T639_beta_raw.pickle'
    dico_save = dict([ 
        ['beta_raw', beta],
        ['nb_lev_beta', nb_lev_beta] 
                    ])
    os.chdir(Osim.path_dir)
    f = open(name_save, 'w')
    pickle.dump(dico_save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time()
    print 'done in {:3.2f} ms'.format(t2-t1)



if COMPUTE_BETA=='phys_smooth':


    #beta = Osim.init_array_spat()
    beta = np.zeros([Osim.nlat, Osim.nlon, nb_lev_beta])

    nb_realisation = np.zeros(nb_lev_beta, int)


    coef_physical_filter1 = 0.03
    coef_physical_filter2 = 0.2

    for day in days:
        for hour in hours:
            ps, Oshell = Osim.load_shell(name_var='ps', day=day, hour=hour)
            if ps.min()==0. :
                print 'problem, ps is full of zeros !'
            else:
                for ip in range(nb_lev_beta):
                    beta[:,:, ip] += H_smooth(ps-plevs[ip]*(1-coef_physical_filter1)
                                                , coef_physical_filter2*plevs[ip])
                    nb_realisation[ip] += 1


    for ip in range(nb_lev_beta):
        beta[:,:,ip] = beta[:,:,ip]/nb_realisation[ip]


    print 'save "beta_phys_smooth"'
    t1 = time()
    name_save = 'AFES_T639_beta_phys_smooth.pickle'
    dico_save = dict([ 
        ['beta_phys_smooth', beta],
        ['nb_lev_beta', nb_lev_beta] 
                    ])
    os.chdir(Osim.path_dir)
    f = open(name_save, 'w')
    pickle.dump(dico_save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time()
    print 'done in {:3.2f} ms'.format(t2-t1)



elif COMPUTE_BETA=='ps_temp_mean':


    ps_temp_mean = np.zeros([Osim.nlat, Osim.nlon])
    nb_realisation = 0


    for day in days:
        for hour in hours:
            ps, Oshell = Osim.load_shell(name_var='ps', day=day, hour=hour)
            if ps.min()==0. :
                print 'problem, ps is full of zeros !'
            else:
                ps_temp_mean += ps
                nb_realisation += 1

    ps_temp_mean = ps_temp_mean/nb_realisation

    print 'save "ps_temp_mean"'
    t1 = time()
    dico_save = dict([ 
        ['ps_temp_mean', ps_temp_mean], 
                    ])
    name_save = Osim.name_simul+'_ps_temp_mean.pickle'
    f = open(Osim.path_dir+'/Statistics/'+name_save, 'w')
    pickle.dump(dico_save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time()
    print 'done in {:3.2f} ms'.format(t2-t1)

    #### plot_field_2D(ps_temp_mean, 'ps_temp_mean', max_field = 1030, min_field = 500)




elif COMPUTE_BETA=='beta_ps_mean':

    print 'load "ps_temp_mean"'
    t1 = time()
    name_save = Osim.name_simul+'_ps_temp_mean.pickle'
    f = open(Osim.path_dir+'/Statistics/'+name_save, 'r')
    dico_save = pickle.load(f)
    f.close()
    ps_temp_mean = dico_save['ps_temp_mean']
    t2 = time()
    print 'done in {:3.2f} ms'.format(t2-t1)


    ps_temp_mean_NF = copy.deepcopy(ps_temp_mean)

    l_filter = 40
    delta_l_filter = 60
    ps_temp_mean_lm = Osim.SH_from_spat(ps_temp_mean)
    ps_temp_mean_lm = ps_temp_mean_lm * H_smooth(l_filter -Osim.l_idx ,delta_l_filter)
    ps_temp_mean = Osim.spat_from_SH(ps_temp_mean_lm)



    beta_ps_mean = np.zeros([nb_lev_beta, Osim.nlat, Osim.nlon])
    coef_physical_filter1 = 0.01

    for ip in range(nb_lev_beta):
        X = (ps_temp_mean-plevs[ip]*(1-coef_physical_filter1) )/plevs[ip]
        deltaX = 0.05
        beta_ps_mean[ip, :,:] = H_smooth(X, deltaX)


    for ip in range(nb_lev_beta):
        l_filter = 15
        delta_l_filter = 60
        beta_temp = beta_ps_mean[ip,:,:]

        beta_lm = Osim.SH_from_spat(beta_temp)
        beta_lm = beta_lm * H_smooth(l_filter -Osim.l_idx ,delta_l_filter)
        beta_temp = Osim.spat_from_SH(beta_lm)
        beta_temp[beta_temp<0.] = 0.
        beta_temp[beta_temp>1.] = 1.

        beta_ps_mean[ip,:,:] = beta_temp


    beta = beta_ps_mean


    print 'save "beta_ps_mean"'
    t1 = time()
    dico_save = dict([ 
        ['beta_ps_mean', beta_ps_mean],
        ['nb_lev_beta', nb_lev_beta] 
                    ])
    name_save = Osim.name_simul+'_beta_ps_mean.pickle'
    f = open(Osim.path_dir+'/Statistics/'+name_save, 'w')
    pickle.dump(dico_save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time()
    print 'done in {:3.2f} ms'.format(t2-t1)





elif COMPUTE_BETA=='tilde':

    print 'load "beta_raw"'
    t1 = time()
    name_save = 'AFES_T639_beta_raw.pickle'
    os.chdir(Osim.path_dir)
    f = open(name_save, 'r')
    dico_save = pickle.load(f)
    f.close()
    beta_raw = dico_save['beta_raw']
    nb_lev_beta = dico_save['nb_lev_beta']
    t2 = time()
    print 'done in {:3.2f} ms'.format(t2-t1)

    beta_tilde = np.zeros([Osim.nlat, Osim.nlon, nb_lev_beta])


    for ip in range(nb_lev_beta):
        l_filter = 12
        delta_l_filter = 40
        beta_temp = beta_raw[:,:,ip]

        beta_lm = Osim.SH_from_spat(beta_temp)
        beta_lm = beta_lm * H_smooth(l_filter -Osim.l_idx ,delta_l_filter)
        beta_temp = Osim.spat_from_SH(beta_lm)
        beta_temp[beta_temp<0.] = 0.
        beta_temp[beta_temp>1.] = 1.

    #    beta_lm = Osim.SH_from_spat(beta_temp)
    #    beta_lm = beta_lm * H_smooth(l_filter -Osim.l_idx ,delta_l_filter)
    #    beta_temp = Osim.spat_from_SH(beta_lm)
    #    beta_temp[beta_temp<0.] = 0.
    #    beta_temp[beta_temp>1.] = 1.
        beta_tilde[:,:,ip] = beta_temp



    #print 'save "beta_tilde"'
    #t1 = time()
    #name_save = 'AFES_T639_beta_tilde.pickle'
    #dico_save = dict([ 
    #    ['beta_tilde', beta_tilde],
    #    ['nb_lev_beta', nb_lev_beta] 
    #                ])
    #os.chdir(Osim.path_dir)
    #f = open(name_save, 'w')
    #pickle.dump(dico_save, f, pickle.HIGHEST_PROTOCOL)
    #f.close()
    #t2 = time()
    #print 'done in {:3.2f} ms'.format(t2-t1)



    #beta = beta_raw
    beta = beta_tilde







LONS = Osim.LONS
LATS = Osim.LATS
m = Basemap(llcrnrlon=0.,llcrnrlat=-90,urcrnrlon=360.,urcrnrlat=90.,\
            resolution='l',area_thresh=10000.,projection='mill')
XX, YY = m(LONS,LATS)


def plot_field_2D(field, for_title='?', max_field=None, min_field=None):
#field = TT
#for_title = 'TT'
#if True:
    hauteur_subfig = 0.87
    largeur_subfig = 0.75
    z_bas_subfig = 0.04
    x_gauche_subplot = 0.08
    fig = plt.figure()
    ax = fig.add_axes([x_gauche_subplot, z_bas_subfig, largeur_subfig, hauteur_subfig])
    mean_field = field.mean()
#    if max_field==None and min_field==None:
#        max_field = abs(field[beta>0.1]).max()
#        min_field = field[beta>0.1].min()
#        max_field = np.ceil(max_field)
#        min_field = np.floor(min_field)
#        if for_title[0]=='h':
#            max_field = 4*(np.sqrt( np.mean(field**2) ))
    nlevel = 30
    if min_field<-0.2:
        levels = -max_field + np.arange(nlevel)*2*max_field/(nlevel-1)
    else:
        levels = min_field + np.arange(nlevel)*(max_field-min_field)/(nlevel-1)
    CS = m.contourf(XX,YY,field,cmap=plt.cm.jet, levels=levels)
    pos = ax.get_position()
    l, b, w, h = pos.bounds
    cax = plt.axes([l+w+0.025, b, 0.025, h]) # setup colorbar axes
    cbar = plt.colorbar(cax=cax) # draw colorbar
    plt.axes(ax)  # make the original axes current again
    plt.hold(True)
    if for_title=='ps' or for_title=='ps_temp_mean':
        print 'plevs =', Osim.plevs[:nb_lev_beta]
        m.contour(XX,YY,field,colors='k', levels=Osim.plevs[:nb_lev_beta])
#    m.drawcoastlines()
    # draw parallels
    delat = 30.
    circles = np.arange(-90.,90.+delat,delat)
    m.drawparallels(circles,labels=[1,0,0,0])
    # draw meridians
    delon = 60.
    meridians = np.arange(60,360,delon)
    m.drawmeridians(meridians,labels=[0,0,0,1])
    plt.title(for_title+'    mean = {:4.2f}'.format(mean_field))
    plt.show()



plot_field_2D(beta[0,:,:], 'beta', max_field = 1.01, min_field = -0.01)

#plot_field_2D(beta[3,:,:], 'beta', max_field = 1.01, min_field = -0.01)



#plot_field_2D(ps_temp_mean, 'ps_temp_mean', max_field = 1040, min_field = 500)


#plot_field_2D(ps_temp_mean_NF, 'ps_temp_mean', max_field = 1040, min_field = 500)


