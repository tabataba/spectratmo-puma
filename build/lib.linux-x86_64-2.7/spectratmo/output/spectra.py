"""Spectra
==========


"""

from spectratmo.output import BaseOutput
from spectratmo.phys_const import R,khi
import numpy as np
from spectratmo.planets import earth
import h5netcdf.legacyapi as net

from netCDF4 import Dataset

class Spectra(BaseOutput):

    _name = 'spectra'


    def compute_1instant(self, instant):
        r"""Compute spectra

        Notes
        -----

        .. math::

           E_A(l,p) = gamma(p) * sum(m):  |theta'_lm (p)|**2 / 2
           E_K(l,p) = sum(m): (u,u)_lm / 2
           C(l,p)   = sum(m): -(omega,alpha)_lm


        Rmq: it would be great to compute the geostrophic and
        ageostrophic spectra!

        """

        ds = self._dataset

        cosLATS = ds.oper.cosLATS
        sinLATS = np.sin(ds.oper.LATS*np.pi/180)
        fCor0 = 2*earth.Omega
        #print('fCor0',fCor0)
        f_LATS = fCor0*sinLATS

        Lambda_p = self.calculate_Lambda()
        #print('TEST1234', ds.make_eddy)
        u3d = ds.get_spatial3dvar('u', instant).astype(np.float64)
        v3d = ds.get_spatial3dvar('v', instant).astype(np.float64)
        o3d = ds.get_spatial3dvar('w', instant).astype(np.float64)
        T3d = ds.get_spatial3dvar('t', instant).astype(np.float64)
        Phi3d = ds.get_spatial3dvar('phi', instant).astype(np.float64)

        #print o3d

        gamma_p = ds.global_tmean.compute_gamma()

        dpu3d = self.partialp_f(u3d)
        dpv3d = self.partialp_f(v3d)

        #E_A,E_K,C
        E_A_pl = np.zeros((ds.nlev,ds.oper.lmax+1))
        E_K_pl = np.zeros((ds.nlev,ds.oper.lmax+1))
        E_K_pl_div = np.zeros((ds.nlev,ds.oper.lmax+1))
        E_K_pl_rot = np.zeros((ds.nlev,ds.oper.lmax+1))
        C_pl = np.zeros((ds.nlev,ds.oper.lmax+1))

        #T_K
        T_Kh_pl = np.zeros((ds.nlev,ds.oper.lmax+1))
        T_Khrot_pl = np.zeros((ds.nlev,ds.oper.lmax+1))
        T_Kv_pl = np.zeros((ds.nlev,ds.oper.lmax+1))

        #F_A
        F_A_pl = np.zeros((ds.nlev,ds.oper.lmax+1))

        #F_K
        F_Kv_pl = np.zeros((ds.nlev,ds.oper.lmax+1))
        F_Kt_pl = np.zeros((ds.nlev,ds.oper.lmax+1))

        #T_A
        T_Ah_pl = np.zeros((ds.nlev,ds.oper.lmax+1))
        T_Av_pl = np.zeros((ds.nlev,ds.oper.lmax+1))

        #L 
        Lcori_pl = np.zeros((ds.nlev,ds.oper.lmax+1))
        Lcalt_pl = np.zeros((ds.nlev,ds.oper.lmax+1))

        #Burgess 2013
        In_pl = np.zeros((ds.nlev,ds.oper.lmax+1))

        T3d_dev = np.empty_like(T3d)
        for ip, p in enumerate(ds.pressure_levels):
            T3d_dev[ip]  = T3d[ip] - self.compute_hmean_representative(T3d[ip], ip)

        #print 't3d'
        #print self.compute_hmean_representative(T3d[ip], ip)
        #print T3d[ip]
        #print T3d_dev[ip]

        # hmean fine
        #print('hi')
        #print(self.compute_hmean_representative(T3d[ip], ip))
        #print(self.compute_hmean(T3d[ip]))
        #print(np.mean(T3d[ip]))

        

        theta3d_dev = np.empty_like(T3d_dev)
        for ip, p in enumerate(ds.pressure_levels):
            theta3d_dev[ip] = Lambda_p[ip] * T3d_dev[ip]

            #print('ip')
            #print(ip)
            #print('Lambda')
            #print(Lambda_p[ip])
            #print('theta3Ddev')
            #print(theta3d_dev[ip])
            dp_T3d_dev = self.partialp_f(T3d_dev)

        #dp_theta3d_dev = self.partialp_f(theta3d_dev) # test

        #ntim=100
        #nlev=10
        #nsp=16512
        #rota=np.loadtxt('/local/home/tabataba/tabataba/runs/diagnostics/rot_1omg.txt')
        #rotb=rota.reshape((ntim,nlev,nsp))
        #diva=np.loadtxt('/local/home/tabataba/tabataba/runs/diagnostics/div_1omg.txt')
        #divb=diva.reshape((ntim,nlev,nsp))

        for ip, p in enumerate(ds.pressure_levels):
            gamma = gamma_p[ip]
            Lambda = Lambda_p[ip]

            #print('ip')
            #print(ip)
            #print('gamma')
            #print(gamma)

            u2d = u3d[ip]
            v2d = v3d[ip]
            o2d = o3d[ip]
            T2d = T3d[ip]
            Phi2d = Phi3d[ip]
            #T2d_hmean   = self.compute_hmean_representative(T2d, ip)
            #T2d_dev     = T2d - T2d_hmean
            #theta2d_dev = Lambda * T2d_dev

            ##theta2d = Lambda * T2d
            ##theta_hmean = self.compute_hmean_representative(theta2d, ip)
            ##theta2d_dev = theta2d - theta_hmean

            #E_A: available potential energy spectrum
            #needs gamma, theta2d_dev

            theta_dev_lm = ds.oper.sh_from_spat(theta3d_dev[ip])
            theta_dev_lm[0] = 0. # was this way in spectatmos. why?

            #E_lm_A = gamma * (np.absolute(theta_dev_lm))**2 / 2.0
            E_A_l = gamma * ds.oper.spectrum_from_sh(theta_dev_lm,'T')
            E_A_pl[ip,:] = E_A_l[:]

            #print "E_A",p,E_A_l

            #E_K: kinetic energy spectrum
            # needs v2d,u2d
            hdiv_lm, hrot_lm = ds.oper.hdivrotsh_from_uv(
                u2d.astype(np.float64), v2d.astype(np.float64)
            )


            #hrot_lm=rotb[instant,ip,0:nsp-1:2]+(rotb[instant,ip,1:nsp:2])*1j

            #hdiv_lm=divb[instant,ip,0:nsp-1:2]+(divb[instant,ip,1:nsp:2])*1j

            uD_lm, uR_lm = ds.oper.uDuRsh_from_hdivrotsh(hdiv_lm, hrot_lm)
            
            #hdiv1_lm = ds.oper.create_array_sh()
            #hrot1_lm = ds.oper.create_array_sh()
            #ds.oper.sh.spat_to_SHsphtor(v2d, u2d, hdiv1_lm, hrot1_lm)

            #print hdiv_lm
            #print hrot_lm
            E_K_l_div=ds.oper.spectrum_from_sh(hdiv_lm, 'hdiv')
            E_K_l_rot=ds.oper.spectrum_from_sh(hrot_lm, 'hrot')
            E_K_l = E_K_l_div + E_K_l_rot
            E_K_pl[ip,:] = E_K_l[:]
            E_K_pl_div[ip,:] = E_K_l_div[:]
            E_K_pl_rot[ip,:] = E_K_l_rot[:]

            #print "E_K",p,E_K_l,E_K_pl_div,E_K_pl_rot

            #C: conversion of APE to KE 
            #needs o2d_lm,T2d_lm,R,p
            #alpha2d = R / p * T2d
            T2d_lm = ds.oper.sh_from_spat(T2d)
            o2d_lm = ds.oper.sh_from_spat(o2d)
            C_l = - ds.oper.cospectrum_from_2fieldssh(o2d_lm, T2d_lm) * R / (p)
            #print R, p
            C_pl[ip,:] = C_l[:]

            #print "C",p,C_l

            #F_A: vertical flux of APE
            otheta_dev_lm = ds.oper.sh_from_spat(o2d * theta3d_dev[ip])
            F_A_l = - gamma * ds.oper.cospectrum_from_2fieldssh(theta_dev_lm, otheta_dev_lm)
            F_A_pl[ip,:] = F_A_l[:]

            #T_A: nonlinear APE spectral flux

            ## compute vertical derivative of Theta
            ## one part is done analytically
            dp_theta_dev = -khi * theta3d_dev[ip]/ds.pressure_levels[ip] + Lambda * dp_T3d_dev[ip] 
            #print(khi)
            #print('khi')

            dp_theta_dev_lm  = ds.oper.sh_from_spat(dp_theta_dev)
            odp_theta_dev_lm = ds.oper.sh_from_spat(o2d * dp_theta_dev)

            T_Av = (
                ds.oper.cospectrum_from_2fieldssh(dp_theta_dev_lm, otheta_dev_lm) -
                ds.oper.cospectrum_from_2fieldssh(theta_dev_lm, odp_theta_dev_lm)
            ) * gamma / 2

            T_Av_pl[ip] = T_Av

            ## horizontal compontent:
            #needs:hdiv_lm, hrot_lm,theta3d_dev,gamma,theta_dev_lm
            hdiv = ds.oper.spat_from_sh(hdiv_lm)
            grad_theta_dev_lon, grad_theta_dev_lat =  ds.oper.gradf_from_fsh(theta_dev_lm)
            temp_AhOp = (
                -u2d * grad_theta_dev_lon - v2d * 
                grad_theta_dev_lat - hdiv * theta3d_dev[ip] /2 )
            temp_AhOp_lm = ds.oper.sh_from_spat(temp_AhOp)
            T_Ah_pl[ip] = ds.oper.cospectrum_from_2fieldssh(
                theta_dev_lm,
                temp_AhOp_lm) * gamma

            #T_K: nonlinear KE spectral flux
            hrot = ds.oper.spat_from_sh(hrot_lm)

            temp_KhOp_lon = -hrot*v2d + hdiv*u2d#/2 #removing this this fixed T_Kh! #removed minus
            temp_KhOp_lat = +hrot*u2d + hdiv*v2d#/2

            hdivtemp_KhOp_lm, hrottemp_KhOp_lm = ds.oper.hdivrotsh_from_uv(
                    temp_KhOp_lon, temp_KhOp_lat)

            temp2_AhOp_lm = ds.oper.sh_from_spat(  
                u2d*u2d + v2d*v2d
                )

            #u2d_lm=ds.oper.sh_from_spat(u2d)
            #v2d_lm=ds.oper.sh_from_spat(v2d)
            #grad_u_lon, grad_u_lat =  ds.oper.gradf_from_fsh(u2d_lm)
            #grad_v_lon, grad_v_lat =  ds.oper.gradf_from_fsh(v2d_lm)
            #tempKhtest_lon = u2d*grad_u_lon+v2d*grad_u_lat
            #tempKhtest_lat = u2d*grad_v_lon+v2d*grad_v_lat
            #hdivtemp_Khtest_lm, hrottemp_Khtest_lm = ds.oper.hdivrotsh_from_uv(
            #        tempKhtest_lon, tempKhtest_lat)           


            hdivdu_lm, hrotdu_lm = ds.oper.hdivrotsh_from_uv(hdiv*u2d, hdiv*v2d)

            T_Kh_pl[ip] = (

                #- ds.oper.cospectrum_from_2divrotsh(
                #    hdiv_lm,            hrot_lm,
                #    hdivtemp_Khtest_lm, hrottemp_Khtest_lm)


                - ds.oper.cospectrum_from_2divrotsh(
                    hdiv_lm,          hrot_lm,
                    hdivtemp_KhOp_lm, hrottemp_KhOp_lm)

                + ds.oper.cospectrum_from_2fieldssh(    #this is the original
                    hdiv_lm, temp2_AhOp_lm
                )/2.0
                #+ ds.oper.cospectrum_from_2divrotsh(      #this is larger (goes to -0.5 in t127 case)
                #    hdiv_lm,          hrot_lm,
                #    hdivdu_lm,        hrotdu_lm
                #)#/2.0
            )

            hdivdpu_lm, hrotdpu_lm = ds.oper.hdivrotsh_from_uv(
                        dpu3d[ip], dpv3d[ip])
            hdivou_lm, hrotou_lm = ds.oper.hdivrotsh_from_uv(o2d*u2d, o2d*v2d)
            hdivodpu_lm, hrotodpu_lm = ds.oper.hdivrotsh_from_uv(
                        o2d*dpu3d[ip], o2d*dpv3d[ip])           

            T_Kv_pl[ip] =  (           
                +ds.oper.cospectrum_from_2divrotsh(
                    hdivdpu_lm, hrotdpu_lm,
                    hdivou_lm,  hrotou_lm)
                -ds.oper.cospectrum_from_2divrotsh(
                    hdiv_lm,     hrot_lm,
                    hdivodpu_lm, hrotodpu_lm)
            )/2


            #hrot = ds.oper.spat_from_sh(hrot_lm)

            #temp_KhOp_lon = -hrot*v2d + hdiv*u2d#/2 #removing this this fixed T_Kh! #removed minus
            #temp_KhOp_lat = +hrot*u2d + hdiv*v2d#/2

            #hdivtemp_KhOp_lm, hrottemp_KhOp_lm = ds.oper.hdivrotsh_from_uv(
            #        temp_KhOp_lon, temp_KhOp_lat)

            #temp2_AhOp_lm = ds.oper.sh_from_spat(  
            #    u2d*u2d + v2d*v2d
            #    )

            #rotation part of T_K
            zeros_lm = ds.oper.create_array_sh(0.)

            urot, vrot = ds.oper.uv_from_hdivrotsh(zeros_lm, hrot_lm)
            temp_KhOp_lon = - hrot * vrot
            temp_KhOp_lat = + hrot * urot
            hdivtemp_KhOp_lm, hrottemp_KhOp_lm = ds.oper.hdivrotsh_from_uv(
                temp_KhOp_lon, temp_KhOp_lat)

            T_Khrot_pl[ip] = -ds.oper.cospectrum_from_2divrotsh(
                zeros_lm,   hrot_lm,
                zeros_lm,   hrottemp_KhOp_lm)

            #rotational KE flux Burgess et al. (2013)
            grad_vort_lon, grad_vort_lat = ds.oper.gradf_from_fsh(hrot_lm)
            vdelvort2 = u2d * grad_vort_lon + v2d * grad_vort_lat
            vdelvort2_lm = ds.oper.sh_from_spat(vdelvort2)

            vort = ds.oper.spat_from_sh(hrot_lm)
            vdelvort = u2d * self.dlam(vort) + v2d * self.dphi(vort)
            vdelvort_lm = ds.oper.sh_from_spat(vdelvort)

            In_pl[ip] = -0.25 * ds.oper.cospectrum_from_2fieldssh2(hrot_lm, vdelvort2_lm)


            #F_K vertical KE flux
            Phi_lm = ds.oper.sh_from_spat(Phi2d)
            F_Kv_pl[ip] = -ds.oper.cospectrum_from_2fieldssh(o2d_lm, Phi_lm)
            hdivou_lm, hrotou_lm = ds.oper.hdivrotsh_from_uv(o2d*u2d, o2d*v2d)

            F_Kt_pl[ip] = -ds.oper.cospectrum_from_2divrotsh(
                hdiv_lm,   hrot_lm,
                hdivou_lm, hrotou_lm
            )/2


            #if instant ==1 and ip == 2:
            #  ntim=100
            #  nlev=10
            #  nsp=16512
            #  rota=np.loadtxt('/local/home/tabataba/tabataba/runs/diagnostics/rot_1omg.txt')
            #  rotb=rota.reshape((ntim,nlev,nsp))
            #  rotc=rotb[instant,ip,0:nsp-1:2]+(rotb[instant,ip,1:nsp:2])*1j

            #  diva=np.loadtxt('/local/home/tabataba/tabataba/runs/diagnostics/div_1omg.txt')
            #  divb=diva.reshape((ntim,nlev,nsp))
            #  divc=divb[instant,ip,0:nsp-1:2]+(divb[instant,ip,1:nsp:2])*1j
            

            #L spectral transfer from coriolis forces
            Fcor_lon = -f_LATS*v2d
            Fcor_lat = +f_LATS*u2d
            hdivFcor_lm, hrotFcor_lm = ds.oper.hdivrotsh_from_uv(
                Fcor_lon, Fcor_lat)
            Lcori_pl[ip] = -ds.oper.cospectrum_from_2divrotsh(
                hdiv_lm,     hrot_lm,
                hdivFcor_lm, hrotFcor_lm)

            psi_lm = ds.oper.create_array_sh(0.)
            chi_lm = ds.oper.create_array_sh(0.)
            COND = ds.oper.l2_idx > 0
            psi_lm[COND] = -earth.radius**2 / ds.oper.l2_idx[COND] * hrot_lm[COND]
            chi_lm[COND] = -earth.radius**2 / ds.oper.l2_idx[COND] * hdiv_lm[COND]
            #print earth.radius
            grad_psi_lon, grad_psi_lat = ds.oper.gradf_from_fsh(psi_lm)
            grad_chi_lon, grad_chi_lat = ds.oper.gradf_from_fsh(chi_lm)
            hdiv1 = ds.oper.spat_from_sh(hdiv_lm)
            hrot1 = ds.oper.spat_from_sh(hrot_lm)
            temp_rot = sinLATS * hdiv1 + cosLATS * grad_chi_lat / earth.radius**2
            temp_div = sinLATS * hrot1 - cosLATS * grad_psi_lat / earth.radius**2
            temp_rot_lm = ds.oper.sh_from_spat(temp_rot)
            temp_div_lm = ds.oper.sh_from_spat(temp_div)
            Lcalt_pl[ip] = fCor0 * (
                ds.oper.cospectrum_from_2fieldssh(psi_lm, temp_rot_lm)
                + ds.oper.cospectrum_from_2fieldssh(chi_lm, temp_div_lm)
            )
            Lcalt_pl[ip,0] = 0

            #if instant == 1 and ip == 2:
            #compare!
            #  f = Dataset('/local/home/tabataba/tabataba/runs/diagnostics/1omg-t127-normtfrc-ac.010-zetad.nc')
            #time,lev,lat,lon
            #  time = f.variables['time']
            #  lat = f.variables['lat']
            #  lon = f.variables['lon']
            #  lev = f.variables['lev']
            #  zeta=f.variables['zeta']
            #  div =f.variables['d']
            #  nlat = lat.shape[0]
            #  nlon = lon.shape[0]
            #  ntime = time.shape[0]
            #  nlev = lev.shape[0]
            #  zeta2d=zeta[instant,ip,:,:]
            #  div2d =div[instant,ip,:,:]
            #  zeta_lm = ds.oper.sh_from_spat(zeta2d[:])
            #  div_lm  = ds.oper.sh_from_spat(div2d[:])


            #ufromspec, vfromspec= ds.oper.uv_from_hdivrotsh(hdiv_lm, hrot_lm)

            #if instant == 1 and ip == 2: print 'comp rot'
            #if instant == 1 and ip == 2: print 'hrot',hrot_lm#, hrot_lm.shape
            #if instant == 1 and ip == 2: print 'rotc',rotc#, rotc.shape
            #if instant ==1 and ip ==2: print 'uR', uR_lm#, uR_lm.shape
            #if instant ==1 and ip ==2: print 'psi', psi_lm#, psi_lm.shape
            #if instant ==1 and ip ==2: print 'zeta_lm', zeta_lm#, zeta_lm.shape
            #if instant == 1 and ip == 2: print 'comp div'
            #if instant == 1 and ip == 2: print 'hdiv', hdiv_lm#, hdiv_lm.shape
            #if instant == 1 and ip == 2: print 'divc', divc#, divc.shape
            #if instant ==1 and ip ==2: print 'uD', uD_lm#, uD_lm.shape
            #if instant ==1 and ip ==2: print 'chi',chi_lm#, chi_lm.shape
            #if instant ==1 and ip ==2: print 'div_lm',div_lm#,div_lm.shape

            #if instant == 1 and ip==2: 
            #  print 'comp u'
            #  print 'u2d',u2d,u2d.shape
            #  print 'ufromspec',ufromspec,ufromspec.shape
            #  print 'comp v'
            #  print 'v2d',v2d,v2d.shape
            #  print 'vfromspec',vfromspec,vfromspec.shape
              #import pdb; pdb.set_trace()
              
            

            #if instant ==0 and ip ==0: print p, uR_lm
            #print instant,ip,hrot1#,hdiv1.shape,hrot1.shape
            #print instant,ip,zeta[instant,ip,:,:]
            #/local/home/tabataba/tabataba/runs/diagnostics/1omg-t127-normtfrc-ac.010-zetad.nc


            #READ IN psi, chi from direct model output!


        return E_A_pl,E_K_pl,C_pl,F_A_pl,T_Av_pl,T_Ah_pl,T_Kh_pl,T_Kv_pl,T_Khrot_pl,F_Kt_pl,F_Kv_pl,Lcori_pl,Lcalt_pl,In_pl
        #raise NotImplementedError

    def compute_tmean(self,):

        ds = self._dataset
        ninstants=np.size(self._dataset.instants)

        E_A_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        E_K_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        C_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        T_Av_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        T_Ah_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        T_Kv_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        T_Kh_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        T_Khrot_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        F_A_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        F_Kt_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        F_Kv_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        Lcori_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        Lcalt_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))
        In_plt = np.zeros((ninstants,ds.nlev,ds.oper.lmax+1))

        for instant in self._dataset.instants:
            E_A,E_K,C,F_A,T_Av,T_Ah,T_Kh,T_Kv,T_Khrot,F_Kt,F_Kv,Lcori,Lcalt,In = (
                ds.spectra.compute_1instant(instant)
                )
            E_A_plt[instant,:] = E_A
            E_K_plt[instant,:] = E_K
            C_plt[instant,:] = C
            F_A_plt[instant,:] = F_A
            T_Av_plt[instant,:] = T_Av
            T_Ah_plt[instant,:] = T_Ah
            T_Kv_plt[instant,:] = T_Kv
            T_Kh_plt[instant,:] = T_Kh
            T_Khrot_plt[instant,:] = T_Khrot
            F_Kt_plt[instant,:] = F_Kt
            F_Kv_plt[instant,:] = F_Kv
            Lcori_plt[instant,:] = Lcori
            Lcalt_plt[instant,:] = Lcalt
            In_plt[instant,:] = In


        self.save_spec2(E_A_plt,varname='E_A_plt')
        self.save_spec2(E_K_plt,varname='E_K_plt')
        self.save_spec2(C_plt,varname='C_plt')
        self.save_spec2(T_Av_plt,varname='T_Av_plt')
        self.save_spec2(T_Ah_plt,varname='T_Ah_plt')
        self.save_spec2(T_Kv_plt,varname='T_Kv_plt')
        self.save_spec2(T_Kh_plt,varname='T_Kh_plt')
        self.save_spec2(T_Khrot_plt,varname='T_Khrot_plt')
        self.save_spec2(F_A_plt,varname='F_A_plt')
        self.save_spec2(F_Kt_plt,varname='F_Kt_plt')
        self.save_spec2(F_Kv_plt,varname='F_Kv_plt')
        self.save_spec2(Lcori_plt,varname='Lcori_plt')
        self.save_spec2(Lcalt_plt,varname='Lcalt_plt')
        self.save_spec2(In_plt,varname='In_plt')

        self.save_spec(np.mean(E_A_plt,axis=0),varname='E_A_tm')
        self.save_spec(np.mean(E_K_plt,axis=0),varname='E_K_tm')
        self.save_spec(np.mean(C_plt,axis=0),varname='C_tm')
        self.save_spec(np.mean(T_Av_plt,axis=0),varname='T_Av_tm')
        self.save_spec(np.mean(T_Ah_plt,axis=0),varname='T_Ah_tm')
        self.save_spec(np.mean(T_Kv_plt,axis=0),varname='T_Kv_tm')
        self.save_spec(np.mean(T_Kh_plt,axis=0),varname='T_Kh_tm')
        self.save_spec(np.mean(T_Khrot_plt,axis=0),varname='T_Khrot_tm')
        self.save_spec(np.mean(F_A_plt,axis=0),varname='F_A_tm')
        self.save_spec(np.mean(F_Kt_plt,axis=0),varname='F_Kt_tm')
        self.save_spec(np.mean(F_Kv_plt,axis=0),varname='F_Kv_tm')
        self.save_spec(np.mean(Lcori_plt,axis=0),varname='Lcori_tm')
        self.save_spec(np.mean(Lcalt_plt,axis=0),varname='Lcalt_tm')
        self.save_spec(np.mean(In_plt,axis=0),varname='In_tm')

            #self.save_spec(E_A,varname='E_A'+'_inst_'+str(instant))
            #self.save_spec(E_K,varname='E_K'+'_inst_'+str(instant))


    def save_spec(self, var, varname='save'):

        ds = self._dataset
        fname = self._get_path_from_name(varname)

        with net.Dataset(fname, 'w') as dsn:
            dsn.createDimension('lev', ds.nlev)
            dsn.createDimension('lmax', ds.oper.lmax+1)
            levs = dsn.createVariable('lev', float, ('lev',))
            levs[:] = ds.pressure_levels
            lmax = dsn.createVariable('lmax', float, ('lmax',))
            lmax[:] = ds.oper.lrange
            v = dsn.createVariable(varname, float, ('lev', 'lmax',))
            v[:] = var

    def save_spec2(self, var, varname='save'):

        ds = self._dataset
        fname = self._get_path_from_name(varname)

        ninstants=np.size(self._dataset.instants)
        #print self._dataset.instants

        with net.Dataset(fname, 'w') as dsn:
            dsn.createDimension('time', ninstants)
            dsn.createDimension('lev', ds.nlev)
            dsn.createDimension('lmax', ds.oper.lmax+1)
            time = dsn.createVariable('time', float, ('time',))
            time[:] = self._dataset.instants
            levs = dsn.createVariable('lev', float, ('lev',))
            levs[:] = ds.pressure_levels
            lmax = dsn.createVariable('lmax', float, ('lmax',))
            lmax[:] = ds.oper.lrange
            v = dsn.createVariable(varname, float, ('time', 'lev', 'lmax',))
            v[:] = var

        #raise NotImplementedError

    def save(self):
        raise NotImplementedError    

    def load(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
