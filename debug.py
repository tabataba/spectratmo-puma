from spectratmo.datasets.puma import PUMADataSet

import numpy as np
import shtns
import pdb
from netCDF4 import Dataset

ds = PUMADataSet()
ds.spectra.compute_tmean()


instant=1
ip=2

print 'easySHT'

u3d = ds.get_spatial3dvar('u', instant).astype(np.float64)
v3d = ds.get_spatial3dvar('v', instant).astype(np.float64)

u2d = u3d[2];v2d = v3d[2]

hdiv_lm, hrot_lm = ds.oper.hdivrotsh_from_uv(
    u2d.astype(np.float64), v2d.astype(np.float64)
)
uuu,vvv=ds.oper.uv_from_hdivrotsh(hdiv_lm,hrot_lm)

print 'uuu',uuu
print 'u2d',u2d
print 'vvv',vvv
print 'v2d',v2d

print 'shtns'


#does not work :(:(:(:(
sh1 = shtns.sht(127, 127, 1, norm=1)
nla,nlo=sh1.set_grid(nlat=192,nphi=384,flags=8708,polar_opt=1e-08, nl_order=2)

radius=6367470.0;l2=sh1.l*(sh1.l+1)

hd = ds.oper.create_array_sh()
hr = ds.oper.create_array_sh()

vx1=-v2d

sh1.spat_to_SHsphtor(vx1, u2d, hd, hr)
hd1=-l2*hd/radius;hr1=l2*hr/radius

COND = l2 > 0
uD_lm=np.empty_like(hd1)
uR_lm=np.empty_like(hr1)
uD_lm[COND] = -hd1[COND] / l2[COND] * radius
uR_lm[COND]  = hr1[COND] / l2[COND] * radius 

#uu=uuu;uu[:]=0
#vv=vvv;vv[:]=0
uu=np.empty_like(uuu)
vv=np.empty_like(vvv)

sh1.SHsphtor_to_spat(uD_lm, uR_lm, vv, uu)

vv=-vv

print 'uu',uu
print 'u2d',u2d
print 'vv',vv
print 'v2d',v2d

#sh1.spat_to_SHsphtor(v2d, u2d, hd, hr)
#sh1.SHsphtor_to_spat(hd, hr, v22, u22)



#this works!!!!
hd2 = ds.oper.create_array_sh()
hr2 = ds.oper.create_array_sh()

sh2 = shtns.sht(127, 127, 1, norm=1)
nla,nlo=sh2.set_grid(nlat=192,nphi=384,flags=8708,polar_opt=1e-08, nl_order=2)

#vx=v2d

sh2.spat_to_SHsphtor(v2d, u2d, hd2, hr2)

hd2=l2*hd2/radius;hr2=l2*hr2/radius

COND = l2 > 0
uD2_lm=np.empty_like(hd2)
uR2_lm=np.empty_like(hr2)

uD2_lm[COND] = hd2[COND] / l2[COND] * radius
uR2_lm[COND]  = hr2[COND] / l2[COND] * radius 
#uD2_lm=hd2
#uR2_lm=hr2

uu2=np.empty_like(uuu)
vv2=np.empty_like(vvv)

sh1.SHsphtor_to_spat(uD2_lm, uR2_lm, vv2, uu2)

print 'uu2',uu2
print 'u2d',u2d
print 'vv2',vv2
print 'v2d',v2d





#this works!!!
T3d = ds.get_spatial3dvar('t', instant).astype(np.float64)
T2d = T3d[2]
T_lm=ds.oper.sh_from_spat(T2d)
ttt=ds.oper.spat_from_sh(T_lm)
#print 'T2d',T2d
#print 'ttt',ttt

ntim=100
nlev=10
nsp=16512
#rota=np.loadtxt('/local/home/tabataba/tabataba/runs/diagnostics/rot_1omg.txt')
#rotb=rota.reshape((ntim,nlev,nsp))
#rotc=rotb[instant,ip,0:nsp-1:2]+(rotb[instant,ip,1:nsp:2])*1j

#diva=np.loadtxt('/local/home/tabataba/tabataba/runs/diagnostics/div_1omg.txt')
#divb=diva.reshape((ntim,nlev,nsp))
#divc=divb[instant,ip,0:nsp-1:2]+(divb[instant,ip,1:nsp:2])*1j

f = Dataset('/local/home/tabataba/tabataba/runs/diagnostics/1omg-t127-normtfrc-ac.010-zetad.nc')
time = f.variables['time']
lat = f.variables['lat']
lon = f.variables['lon']
lev = f.variables['lev']
zeta=f.variables['zeta']
div =f.variables['d']
nlat = lat.shape[0]
nlon = lon.shape[0]
ntime = time.shape[0]
nlev = lev.shape[0]
zeta2d=zeta[instant,ip,:,:]
div2d =div[instant,ip,:,:]
zeta_lm = ds.oper.sh_from_spat(zeta2d[:])
div_lm  = ds.oper.sh_from_spat(div2d[:])

#print 'divc',divc
print 'hdiv',hdiv_lm
print 'hd1',hd1
print 'hd2',hd2
print 'div',div_lm

#print 'rotc',rotc
print 'hrot',hrot_lm
print 'hr1',hr1
print 'hr2',hr2
print 'rot',zeta_lm

