#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# to use for example with ipython -pylab
# run /home/pierre/Python/Production/Energy_budget/all_terms_loop_days_review.py
# run /scratch/augier/Python/Production/Energy_budget/all_terms_loop_days_review.py

# compute some terms of the spectral energy budget.
# The memory is used just as needed.

# import basic modules...
import os, sys, resource
import numpy as np
# import homemade module to load data and perform spharm transform
import treat_simul_shtns22 as treat_simul
# import function time in module time for timming functions
from time import time
# import module for saving...
####import pickle
import cPickle as pickle

import matplotlib.pyplot as plt




def cumsum_inv(a):
    return a[::-1].cumsum()[::-1]

SAVE = 1

season = 'summer'
#season = 'winter'


name_simul = 'AFES_T639'
#name_simul = 'AFES_T1279'
name_simul = 'ECMWF_T1279'
#name_simul = 'ECMWF_T159'
Osim = treat_simul.treat_simul(name_simul=name_simul)


####print resource.getrusage(resource.RUSAGE_SELF).ru_maxrss



plevs = Osim.plevs
nz = len(plevs)
p00 = 1000   # hPa
A = (p00/plevs)**Osim.khi

cosLATS = Osim.cosLATS
sinLATS = np.sin(Osim.LATS*np.pi/180)
fCor0 = 2*Osim.Omega
f_LATS = fCor0*sinLATS

beta3D_small, nb_lev_beta = Osim.load_beta_ps_mean()
beta3D = np.ones([nz, Osim.nlat, Osim.nlon], dtype=np.float32)
beta3D[0:nb_lev_beta] = beta3D_small




Osim.load_meanT_coefAPE(season=season)
Coef_APE_Theta = Osim.Coef_APE_Theta


zeros_lm = Osim.init_array_SH(0.)


# For these files, we use the notations of the resubmitted version of the paper
# A new formulation of the spectral energy budget of the atmosphere...

# the tendencies 
P_TKhOp_l = np.zeros([nz,Osim.lmax+1])
P_TKvOp_l = np.zeros([nz,Osim.lmax+1])
P_TKrot_l = np.zeros([nz,Osim.lmax+1])
P_TAhOp_l = np.zeros([nz,Osim.lmax+1])
P_TAvOp_l = np.zeros([nz,Osim.lmax+1])
P_Lcori_l = np.zeros([nz,Osim.lmax+1])
P_Lcalt_l = np.zeros([nz,Osim.lmax+1])
P_Conv__l = np.zeros([nz,Osim.lmax+1])
P_Conv2_l = np.zeros([nz,Osim.lmax+1])
P_DKh___l = np.zeros([nz,Osim.lmax+1])

# the vertical fluxes
F_Kpres_l = np.zeros([nz,Osim.lmax+1])
F_Kturb_l = np.zeros([nz,Osim.lmax+1])
F_Aturb_l = np.zeros([nz,Osim.lmax+1])



list_ens_tendencies = []
for ip in range(nz):
    ens_tendencies = treat_simul.ensemble_variables('ensemble tendency terms')
    ens_tendencies.ddata['P_TKhOp_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_TKvOp_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_TKrot_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_TAhOp_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_TAvOp_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_Lcori_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_Lcalt_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_Conv__l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_Conv2_l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_DKh___l'] = np.zeros([Osim.lmax+1])
    ens_tendencies.ddata['P_Conv_'] = 0.
    ens_tendencies.ddata['P_DKh__'] = 0.
    list_ens_tendencies.append(ens_tendencies)

list_ens_vert_fluxes = []
for ip in range(nz):
    ens_vert_fluxes = treat_simul.ensemble_variables('ensemble vertical fluxes terms')
    ens_vert_fluxes.ddata['F_Kpres_l'] = np.zeros([Osim.lmax+1])
    ens_vert_fluxes.ddata['F_Kturb_l'] = np.zeros([Osim.lmax+1])
    ens_vert_fluxes.ddata['F_Aturb_l'] = np.zeros([Osim.lmax+1])
    list_ens_vert_fluxes.append(ens_vert_fluxes)

def hour_max(day):
    if day==11:
        h = 8
    else:
        h = 24
    return h



delta_hour = 2
if name_simul[:4]=='AFES':
    if Osim.computer=='pierre-KTH':
        days = [5,11,15]
    elif Osim.computer=='KTH':
        days = [4,5,8,11,12,13,14,15]
elif name_simul=='ECMWF_T1279':
    days = [1, 5, 10, 15, 20, 25]
elif name_simul=='ECMWF_T159':
    days = [5]

#delta_hour = 24
#days = [4,5,8,11]
#days = [4]

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
    for ip in range(nz):
        list_ens_tendencies[ip].reset_to_zeros()
        list_ens_vert_fluxes[ip].reset_to_zeros()


    hours = hours_day_namesimul(day, name_simul)
    nb_instants_computed_day = 0
    for hour in hours:
        t1_1time = time()

        # For each time, we compute (in this order) the values of:
        # F_Kpres_l[ip, il], P_Conv__l[ip, il]
        # F_Aturb_l[ip, il], P_TAvOp_l[ip, il], P_Conv2_l[ip, il]
        # P_TAhOp_l[ip, il]
        # P_TKhOp_l[ip, il], P_TKrot_l[ip, il], P_DKh___l[ip, il]
        # P_TKvOp_l[ip, il], F_Kturb_l[ip, il]
        # P_Lcori_l[ip, il]


        # We first list the things that we have to do:

        # load omega3D, Phi3D, TT3D
        # compute omegab3D_lm, Phib3D_lm, TTb3D_lm
        # del(Phib3D, oob3D)
        # compute   F_Kpres_l[ip, il], 
        #           P_Conv__l[ip, il]
        # del(Phib3D_lm), del(Tb3D_lm)

        # compute Thetabp3D from T3D, then Thetabp3D_lm
        # del(T3D)
        # compute dp_Thetabp3D
        # compute   F_Aturb_l[ip, il], 
        #           P_TAvOp_l[ip, il]
        # del(dp_Thetabp3D)

        # load uu3D, vv3D
        # compute d3D
        # compute P_TAhOp_l[ip, il]
        # del(Thetabp3D, Thetabp3D_lm, d3D)

        # compute uub3D, vvb3D
        # del(uu3D, vv3D)
        # compute divhuub3D_lm, rothuub3D_lm
        # compute   P_TKhOp_l[ip, il], 
        #           P_TKrot_l[ip, il], 
        #           P_DKh___l[ip, il]

        # compute dp_uub3D
        # P_TKvOp_l[ip, il], F_Kturb_l[ip, il]
        # del(omega3D, dp_uub3D)

        # compute   P_Lcori_l[ip, il]
        #           P_Lcalt_l[ip, il]
        # del(uub3D, vvb3D, divhuub3D_lm, rothuub3D_lm)




        # Then, the computations:

        # load oo3D, Phi3D, T3D
        print 'load oo3D, Phi3D, T3D'
        t1 = time()
        oo3D  = Osim.load_var3D(name_var='o',   day=day, hour=hour, season=season) # en Pa/s
        Phi3D = Osim.load_var3D(name_var='Phi', day=day, hour=hour, season=season)
        TT3D  = Osim.load_var3D(name_var='T',   day=day, hour=hour, season=season)
        t2 = time()
        print '(loaded in {0:3.2f} s)'.format(t2-t1)

        Phib3D = Phi3D
        Phib3D[0:nb_lev_beta] = Phi3D[0:nb_lev_beta] * beta3D_small
        del(Phi3D)

        TTb3D = np.empty(TT3D.shape)
        TTb3D[nb_lev_beta:nz] = TT3D[nb_lev_beta:nz]
        TTb3D[0:nb_lev_beta]  = TT3D[0:nb_lev_beta] * beta3D_small


        print '1 SH3D transform, compute oob3D_lm',
        sys.stdout.flush()
        t1 = time()
        oob3D_lm = np.empty([nz, Osim.nlm], dtype=complex)
        for ip in range(nz):
            oob3D_lm[ip] = Osim.SH_from_spat(oo3D[ip]*beta3D[ip])
            # we use an analytical result for levels above the surface...
            if ip>=nb_lev_beta:
                oob3D_lm[ip, 0] = 0.
                oo3D[ip] = Osim.spat_from_SH(oob3D_lm[ip])
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)

        # compute Phib3D_lm, TTb3D_lm
        print'2 SH3D transforms, compute Phib3D_lm, TTb3D_lm',
        sys.stdout.flush()
        t1 = time()
        Phib3D_lm = np.zeros([nz, Osim.nlm], dtype=complex)
        TTb3D_lm =   np.zeros([nz, Osim.nlm], dtype=complex)
        for ip in range(nz):
            Phib3D_lm[ip] = Osim.SH_from_spat(Phib3D[ip])
            TTb3D_lm[ip] =   Osim.SH_from_spat(TTb3D[ip])
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)

        # del(Phib3D)
        del(Phib3D)


        # compute   F_Kpres_l[ip, il], 
        #           P_Conv__l[ip, il]
        print 'compute F_Kpres_l, P_Conv__l',
        sys.stdout.flush()
        t1 = time()
        for ip in range(nz):
            # spectrum of vertical pressure flux
            F_Kpres_l[ip] = -Osim.cospectrum_from_2fieldsSH(oob3D_lm[ip], 
                                                            Phib3D_lm[ip])

            # spectrum of conversion
            P_Conv__l[ip] = -Osim.cospectrum_from_2fieldsSH(oob3D_lm[ip], 
                                                            TTb3D_lm[ip])*Osim.R/(plevs[ip]*100)
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)


        # del(Phib3D_lm), del(TTb3D_lm)
        del(Phib3D_lm)
        del(TTb3D_lm)

        # compute Thetabp3D from T3D, then Thetabp3D_lm from TTb3D
        print 'compute Thetabp3D and dp_Thetabp3D',
        sys.stdout.flush()
        t1 = time()

        TTbp3D = np.empty([nz, Osim.nlat, Osim.nlon])
        for ip in range(nz):
            TTbp3D[ip] = (TT3D[ip]-Osim.mean_field_representative(TT3D[ip], ip))*beta3D[ip]
        del(TT3D)

        Thetabp3D = np.empty([nz, Osim.nlat, Osim.nlon])
        for ip in range(nz):
            Thetabp3D[ip] = A[ip]*TTbp3D[ip]


        # compute dp_Thetabp3D
        ## compute vertical derivative of Theta
        ## one part is done analytically
        dp_TTbp3D = Osim.vertical_derivative_f(TTbp3D)
        del(TTbp3D)

        dp_Thetabp3D = np.empty([nz, Osim.nlat, Osim.nlon])
        for ip in range(nz):
            dp_Thetabp3D[ip] = -Osim.khi*Thetabp3D[ip]/plevs[ip] + A[ip]*dp_TTbp3D[ip]
        del(dp_TTbp3D)

        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)



        # compute Thetabp3D_lm
        print'1 SH3D transforms, compute Thetabp3D_lm',
        sys.stdout.flush()
        t1 = time()
        Thetabp3D_lm = np.zeros([nz, Osim.nlm], dtype=complex)
        for ip in range(nz):
            Thetabp3D_lm[ip] = Osim.SH_from_spat(Thetabp3D[ip])
            Thetabp3D_lm[ip,0] = 0.
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)

        # compute   F_Aturb_l[ip, il], 
        #           P_TAvOp_l[ip, il]
        print 'compute F_Aturb_l, P_TAvOp_l, P_Conv2_l',
        sys.stdout.flush()
        t1 = time()
        for ip in range(nz):
            dp_Thetabp_lm = Osim.SH_from_spat(dp_Thetabp3D[ip])
            ooThetabp_lm = Osim.SH_from_spat(oo3D[ip]*Thetabp3D[ip])
            oodp_Thetabp_lm = Osim.SH_from_spat(oo3D[ip]*dp_Thetabp3D[ip])

            F_Aturb_l[ip] = -Osim.cospectrum_from_2fieldsSH(
                                                            Thetabp3D_lm[ip], 
                                                            ooThetabp_lm
                                            )*Coef_APE_Theta[ip]/2

            P_TAvOp_l[ip] =  (  +Osim.cospectrum_from_2fieldsSH(
                                                    dp_Thetabp_lm, ooThetabp_lm)
                                -Osim.cospectrum_from_2fieldsSH(
                                                    Thetabp3D_lm[ip], oodp_Thetabp_lm)
                                            )*Coef_APE_Theta[ip]/2/100



            TTbp_lm = Thetabp3D_lm[ip]/A[ip]
            P_Conv2_l[ip] = -Osim.cospectrum_from_2fieldsSH(oob3D_lm[ip], 
                                                            TTbp_lm)*Osim.R/(plevs[ip]*100)

        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)
        del(dp_Thetabp3D)


        # load uu3D, vv3D
        print 'load uu3D, vv3D'
        t1 = time()
        uu3D = Osim.load_var3D(name_var='u', day=day, hour=hour, season=season)
        vv3D = Osim.load_var3D(name_var='v', day=day, hour=hour, season=season)
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)



        # compute P_TAhOp_l[ip, il]
        print 'compute P_TAhOp_l',
        sys.stdout.flush()
        t1 = time()
        for ip in range(nz):
            hdiv_lm, hrot_lm = Osim.hdivrotSH_from_uuvv(uu3D[ip], vv3D[ip])
            hdiv = Osim.spat_from_SH(hdiv_lm)
            grad_Thetabp_lon, grad_Thetabp_lat = Osim.gradf_from_fSH(Thetabp3D_lm[ip])
            temp_AhOp = (   -uu3D[ip]*grad_Thetabp_lon - vv3D[ip]*grad_Thetabp_lat
                            -hdiv*Thetabp3D[ip]/2 )
            temp_AhOp_lm= Osim.SH_from_spat(temp_AhOp)
            P_TAhOp_l[ip] = Osim.cospectrum_from_2fieldsSH( Thetabp3D_lm[ip], 
                                                        temp_AhOp_lm
                                                )*Osim.Coef_APE_Theta[ip]
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)

        # del(Thetabp3D, Thetabp3D_lm)
        del(Thetabp3D)
        del(Thetabp3D_lm)



        # compute uub3D, vvb3D and del(uu3D, vv3D)
        uub3D = uu3D
        uub3D[0:nb_lev_beta] = uu3D[0:nb_lev_beta] * beta3D_small
        del(uu3D)
        vvb3D = vv3D
        vvb3D[0:nb_lev_beta] = vv3D[0:nb_lev_beta] * beta3D_small
        del(vv3D)

        # compute divhuub3D_lm, rothuub3D_lm
        print'1 vectorial SH3D transform, compute divhuub3D_lm, rothuub3D_lm',
        sys.stdout.flush()
        t1 = time()
        divhuub3D_lm = np.zeros([nz, Osim.nlm], dtype=complex)
        rothuub3D_lm =   np.zeros([nz, Osim.nlm], dtype=complex)
        for ip in range(nz):
            divhuub3D_lm[ip], rothuub3D_lm[ip] = Osim.hdivrotSH_from_uuvv(
                                                            uub3D[ip], vvb3D[ip])
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)




        # compute dp_uub3D and dp_vvb3D
        print 'compute dp_uub3D and dp_vvb3D',
        sys.stdout.flush()
        t1 = time()
        dp_uub3D = Osim.vertical_derivative_f(uub3D)
        dp_vvb3D = Osim.vertical_derivative_f(vvb3D)
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)

        # compute P_TKvOp_l, F_Kturb_l
        print 'compute P_TKvOp_l, F_Kturb_l',
        sys.stdout.flush()
        t1 = time()
        for ip in range(nz):

            divhdp_uub_lm, rothdp_uub_lm = Osim.hdivrotSH_from_uuvv(
                        dp_uub3D[ip], dp_vvb3D[ip])

            divhoouub_lm, rothoouub_lm = Osim.hdivrotSH_from_uuvv(
                        oo3D[ip]*uub3D[ip], oo3D[ip]*vvb3D[ip])

            divhoodp_uub_lm, rothoodp_uub_lm = Osim.hdivrotSH_from_uuvv(
                        oo3D[ip]*dp_uub3D[ip], oo3D[ip]*dp_vvb3D[ip])

            F_Kturb_l[ip] = -Osim.cospectrum_from_2divrotSH(
                                            divhuub3D_lm[ip],   rothuub3D_lm[ip],
                                            divhoouub_lm,       rothoouub_lm
                                            )/2

            P_TKvOp_l[ip] =  (  +Osim.cospectrum_from_2divrotSH(
                                            divhdp_uub_lm,      rothdp_uub_lm,
                                            divhoouub_lm,       rothoouub_lm)
                                -Osim.cospectrum_from_2divrotSH(
                                            divhuub3D_lm[ip],   rothuub3D_lm[ip],
                                            divhoodp_uub_lm,    rothoodp_uub_lm)
                                )/2/100
        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)
        del(dp_uub3D)
        del(dp_vvb3D)
        del(oo3D)



        # compute   P_TKhOp_l[ip, il], 
        #           P_TKrot_l[ip, il], 
        #           P_DKh___l[ip, il]

        print 'compute P_TKhOp_l, P_TKrot_l, P_DKh___l',
        sys.stdout.flush()
        t1 = time()
        for ip in range(nz):

            rothuub = Osim.spat_from_SH(rothuub3D_lm[ip])
            divhuub = Osim.spat_from_SH(divhuub3D_lm[ip])

            temp_KhOp_lon = -rothuub*vvb3D[ip] + divhuub*uub3D[ip]/2
            temp_KhOp_lat = +rothuub*uub3D[ip] + divhuub*vvb3D[ip]/2

            divhtemp_KhOp_lm, rothtemp_KhOp_lm = Osim.hdivrotSH_from_uuvv(
                    temp_KhOp_lon, temp_KhOp_lat)

            temp2_AhOp_lm = Osim.SH_from_spat(  uub3D[ip]*uub3D[ip]
                                                +vvb3D[ip]*vvb3D[ip])

            P_TKhOp_l[ip] = (   
                                -Osim.cospectrum_from_2divrotSH(
                                                divhuub3D_lm[ip],   rothuub3D_lm[ip],
                                                divhtemp_KhOp_lm,   rothtemp_KhOp_lm)
                                +Osim.cospectrum_from_2fieldsSH( 
                                                divhuub3D_lm[ip], 
                                                temp2_AhOp_lm
                                                    )/2
                            )


            uub_rot, vvb_rot = Osim.uuvv_from_hdivrotSH(zeros_lm, rothuub3D_lm[ip])
            temp_KhOp_lon = -rothuub*vvb_rot
            temp_KhOp_lat = +rothuub*uub_rot
            divhtemp_KhOp_lm, rothtemp_KhOp_lm = Osim.hdivrotSH_from_uuvv(
                    temp_KhOp_lon, temp_KhOp_lat)
            P_TKrot_l[ip] = -Osim.cospectrum_from_2divrotSH(
                                                zeros_lm,   rothuub3D_lm[ip],
                                                zeros_lm,   rothtemp_KhOp_lm)



        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)







        # compute   P_Lcori_l[ip, il]
        #           P_Lcalt_l[ip, il]

        print 'compute P_Lcori_l, P_Lcalt_l',
        sys.stdout.flush()
        t1 = time()
        for ip in range(nz):
            Fcor_lon = -f_LATS*vvb3D[ip]
            Fcor_lat = +f_LATS*uub3D[ip]
            divhFcor_lm, rothFcor_lm = Osim.hdivrotSH_from_uuvv(
                            Fcor_lon, Fcor_lat)
            P_Lcori_l[ip] = -Osim.cospectrum_from_2divrotSH(
                                                divhuub3D_lm[ip],   rothuub3D_lm[ip],
                                                divhFcor_lm,        rothFcor_lm)


            psi_lm = -Osim.r_a**2/Osim.l2_idx*rothuub3D_lm[ip]
            chi_lm = -Osim.r_a**2/Osim.l2_idx*divhuub3D_lm[ip]
            grad_psi_lon, grad_psi_lat = Osim.gradf_from_fSH(psi_lm)
            grad_chi_lon, grad_chi_lat = Osim.gradf_from_fSH(chi_lm)
            divhuub = Osim.spat_from_SH(divhuub3D_lm[ip])
            rothuub = Osim.spat_from_SH(rothuub3D_lm[ip])
            temp_rot = sinLATS*divhuub + cosLATS*grad_chi_lat/Osim.r_a
            temp_div = sinLATS*rothuub + cosLATS*grad_psi_lat/Osim.r_a
            temp_rot_lm = Osim.SH_from_spat(temp_rot)
            temp_div_lm = Osim.SH_from_spat(temp_div)
            P_Lcalt_l[ip] = fCor0*(   Osim.cospectrum_from_2fieldsSH(psi_lm, temp_rot_lm)
                        - Osim.cospectrum_from_2fieldsSH(chi_lm, temp_div_lm)
                    )

        t2 = time()
        print '(done in {0:3.2f} s)'.format(t2-t1)


        # del(uub3D, vvb3D, divhuub3D_lm, rothuub3D_lm)
        del(uub3D)
        del(vvb3D)
        del(divhuub3D_lm)
        del(rothuub3D_lm)




        print 'build the ensemble and add for the time average'

        for ip in range(nz):
            ens_tendencies = treat_simul.ensemble_variables('ensemble tendency terms')
            ens_tendencies.ddata['P_TKhOp_l'] = P_TKhOp_l[ip]
            ens_tendencies.ddata['P_TKvOp_l'] = P_TKvOp_l[ip]
            ens_tendencies.ddata['P_TKrot_l'] = P_TKrot_l[ip]
            ens_tendencies.ddata['P_TAhOp_l'] = P_TAhOp_l[ip]
            ens_tendencies.ddata['P_TAvOp_l'] = P_TAvOp_l[ip]
            ens_tendencies.ddata['P_Lcori_l'] = P_Lcori_l[ip]
            ens_tendencies.ddata['P_Lcalt_l'] = P_Lcalt_l[ip]
            ens_tendencies.ddata['P_Conv__l'] = P_Conv__l[ip]
            ens_tendencies.ddata['P_Conv2_l'] = P_Conv2_l[ip]
            ens_tendencies.ddata['P_DKh___l'] = P_DKh___l[ip]
            ens_tendencies.ddata['P_Conv_']   = P_Conv__l[ip].sum()
            ens_tendencies.ddata['P_DKh__']   = P_DKh___l[ip].sum()
            list_ens_tendencies[ip]    += ens_tendencies
            ens_vert_fluxes = treat_simul.ensemble_variables('ensemble vertical fluxes terms')
            ens_vert_fluxes.ddata['F_Kpres_l'] = F_Kpres_l[ip]
            ens_vert_fluxes.ddata['F_Kturb_l'] = F_Kturb_l[ip]
            ens_vert_fluxes.ddata['F_Aturb_l'] = F_Aturb_l[ip]
            list_ens_vert_fluxes[ip]   += ens_vert_fluxes

        nb_instants_computed_day += 1
        nb_instants_computed += 1
        nb_to_do = nb_to_do - 1

        t2_1time = time()
        print '1 time treated in {0:3.2f} s'.format(t2_1time-t1_1time)

        if nb_instants_computed%(nbtot/100.)<1.:
            print 'day =', day, 'hour =', hour, ' completed: {0:3.0f}% done'.format(nb_instants_computed/float(nbtot)*100.)

        if not nb_to_do==0:
            print 'there are still {0} instants to treat'.format(nb_to_do)
            print 'approximative time left: {0:5.0f} s'.format((t2_1time-t1_1time)*nb_to_do)
        else:
            print 'computation completed...'


    for ip in range(nz):
        list_ens_tendencies[ip] = list_ens_tendencies[ip]/nb_instants_computed_day



    # il faut calculer la liste list_ens_cumul_tend
    list_ens_cumul_tend = []
    for ip in range(nz):
        ens_tendencies   = list_ens_tendencies[ip]
        ens_vert_fluxes  = list_ens_vert_fluxes[ip]

        Pi_TKhOp_l = cumsum_inv(ens_tendencies.ddata['P_TKhOp_l'])
        Pi_TKvOp_l = cumsum_inv(ens_tendencies.ddata['P_TKvOp_l'])
        Pi_TKrot_l = cumsum_inv(ens_tendencies.ddata['P_TKrot_l'])
        Pi_TAhOp_l = cumsum_inv(ens_tendencies.ddata['P_TAhOp_l'])
        Pi_TAvOp_l = cumsum_inv(ens_tendencies.ddata['P_TAvOp_l'])
        Pi_Lcori_l = cumsum_inv(ens_tendencies.ddata['P_Lcori_l'])
        Pi_Lcalt_l = cumsum_inv(ens_tendencies.ddata['P_Lcalt_l'])
        cumu_Con_l = cumsum_inv(ens_tendencies.ddata['P_Conv__l'])
        cumu_Co2_l = cumsum_inv(ens_tendencies.ddata['P_Conv2_l'])
        cumu_DKh_l = cumsum_inv(ens_tendencies.ddata['P_DKh___l'])
        cumu_FKp_l = cumsum_inv(ens_vert_fluxes.ddata['F_Kpres_l'])
        cumu_FKt_l = cumsum_inv(ens_vert_fluxes.ddata['F_Kturb_l'])
        cumu_FAt_l = cumsum_inv(ens_vert_fluxes.ddata['F_Aturb_l'])

        ens_cumul_tend = treat_simul.ensemble_variables('ensemble flux terms')
        ens_cumul_tend.ddata['Pi_TKhOp_l'] = Pi_TKhOp_l
        ens_cumul_tend.ddata['Pi_TKvOp_l'] = Pi_TKvOp_l
        ens_cumul_tend.ddata['Pi_TKrot_l'] = Pi_TKrot_l
        ens_cumul_tend.ddata['Pi_TAhOp_l'] = Pi_TAhOp_l
        ens_cumul_tend.ddata['Pi_TAvOp_l'] = Pi_TAvOp_l
        ens_cumul_tend.ddata['Pi_Lcori_l'] = Pi_Lcori_l
        ens_cumul_tend.ddata['Pi_Lcalt_l'] = Pi_Lcalt_l
        ens_cumul_tend.ddata['cumu_Con_l'] = cumu_Con_l
        ens_cumul_tend.ddata['cumu_Co2_l'] = cumu_Co2_l
        ens_cumul_tend.ddata['cumu_DKh_l'] = cumu_DKh_l
        ens_cumul_tend.ddata['cumu_FKp_l'] = cumu_FKp_l
        ens_cumul_tend.ddata['cumu_FKt_l'] = cumu_FKt_l
        ens_cumul_tend.ddata['cumu_FAt_l'] = cumu_FAt_l
        list_ens_cumul_tend.append(ens_cumul_tend)

    if SAVE:
        name_directory_save = Osim.path_dir+'/Statistics/Dyn_days_review'
        if name_simul=='ECMWF_T1279':
            name_directory_save = name_directory_save+season

        if not os.path.exists(name_directory_save):
            os.mkdir(name_directory_save)
        name_save = 'T'+str(Osim.lmax)+'_dyn_day'+str(day)+'.pickle'
        dico_save = dict([  ['list_ens_tendencies',  list_ens_tendencies],
                            ['list_ens_vert_fluxes', list_ens_vert_fluxes],
                            ['list_ens_cumul_tend',  list_ens_cumul_tend],
                            ['name_save', name_save]
                        ])
        f = open(name_directory_save+'/'+name_save, 'w')
        pickle.dump(dico_save, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    print '\nday = '+str(day)+' is done'
    print 'nb_instants_computed_day =', nb_instants_computed_day
    print '\n'






