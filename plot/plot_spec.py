#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.colors as colors

import sys

def cumsum_inv(a):
    return a[::-1].cumsum()[::-1]
#    return a

from argparse import ArgumentParser
parser = ArgumentParser()


from netCDF4 import Dataset
#>>> f = MFDataset('mftest*nc')
#>>> print f.variables['x'][:]

parser.add_argument("-n","--name",action="store",type=str)
parser.add_argument("-p","--path",action="store",type=str)
parser.add_argument("-e","--eddypath",action="store",type=str)
parser.add_argument("-o","--output",action="store",type=str)  

args, unknown = parser.parse_known_args()

a = 6400000.0#/8.0
g = 9.8
kappa = 0.286
M_air = 28.98
k = 1.38E-23
R=8.3144621
p0=100000

ncfiles=['E_A_tm.nc','E_K_tm.nc','C_tm.nc','T_Ah_tm.nc','T_Av_tm.nc','T_Kh_tm.nc','T_Kv_tm.nc','T_Khrot_tm.nc','F_A_tm.nc','F_Kt_tm.nc','F_Kv_tm.nc','Lcalt_tm.nc','Lcori_tm.nc','In_tm.nc']
ncnames=['E_A_tm','E_K_tm','C_tm','T_Ah_tm','T_Av_tm','T_Kh_tm','T_Kv_tm','T_Khrot_tm','F_A_tm','F_Kt_tm','F_Kv_tm','Lcalt_tm','Lcori_tm','In_tm']

#f=[]*13
#f=list()

print len(ncnames)

f = Dataset(args.path+'/'+ncfiles[0])
aa=f.variables[ncnames[0]]
lmax=f.variables['lmax']
lsize=len(lmax[:])


fig, axarr = plt.subplots(4, 4, sharex='row', figsize=(3*10,4*10))

axarr = axarr.ravel()

cc_sav=np.empty((len(ncnames),lsize))

for i in range(0,len(ncnames)):
  ax=axarr[i]
  print ax
  f = Dataset(args.path+'/'+ncfiles[i])
  aa=f.variables[ncnames[i]]
  lmax=f.variables['lmax']
  #print aa[:]
  bb=aa[:]
  #dp = p0/10. / g
  #for i in range(140):
  #  print cc[i,0],i
  plotx=[[]]
  cc = np.zeros_like(bb)
  if i == 8 or i==9 or i==10:
    dp = 1
  else: 
    dp = p0/10. / g
  for j in range(10):
    if i == 0 or i == 1:
      cc[j,:]= bb[j,:] * dp
    else:
      cc[j,:]= cumsum_inv(bb[j,:]) * dp
    plotx = plotx + [ax.plot(lmax,cc[j,:], label=j)]
  if i==0: ax.legend()
  #if i==13:
  #  cc_intp = np.mean(cc,axis=0)
  #else:
  cc_intp = np.sum(cc,axis=0)
  cc_sav[i,:]=cc_intp[:]
  if i == 0 or i == 1:
    plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-5.))),linewidth=3.0,color='k')
    plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-5./3.))),linewidth=3.0,color='k')
    plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-3.))),linewidth=3.0,color='k')
  #print(lmax[1::])
  #print(1E7*(pow(lmax[1::],(-5./3.))))
  plot = ax.plot(lmax,cc_intp,linewidth=4.0,color='b')
  ax.set_xscale('log')
  if i < 2: ax.set_yscale('log')
  #if i == 8 or i==9 or i==10: 
  ax.set_ylabel('l')
  ax.set_ylabel(ncnames[i])


plt.tight_layout()
outname=args.output+'/'+args.name
plt.savefig(outname+'.pdf',format='pdf',dpi=400)


colors=['k','DarkOrange','b','g','DarkOrange','DarkOrange']
linestyles=['-','-','-','-.',':','--']
linewidths=[4,2,2,2,1.5,1.5]
labels=[r'$\Pi$',r'$\Pi_K$',r'$\Pi_A$',r'$\mathcal{C}$',r'$\Pi_{Krot}$',r'$\Pi_{Kdiv}$']

############ EOF

#0  E_A
#1  E_K
#2  C
#3  T_Ah
#4  T_Av
#5  T_Kh
#6  T_Kv
#7  T_Khrot
#8  F_A
#9  F_Kt
#10 F_Kv
#11 Lcalt
#12 Lcori 
#13 Burgess flux

cc_plot=np.empty((6,lsize))
#C
cc_plot[3,:] = cc_sav[2,:]
#Pi_A
cc_plot[2,:] = cc_sav[3,:] + cc_sav[4,:]
#Pi_K
cc_plot[1,:] = cc_sav[5,:] + cc_sav[6,:]
#Pi_tot
cc_plot[0,:] = cc_plot[1,:] + cc_plot[2,:]

#Pi_Krot
cc_plot[4,:] = cc_sav[7,:]
#Pi_Kdiv
cc_plot[5,:] = cc_plot[1,:] - cc_plot[4,:]

fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(1.5*10,1.0*10))

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
#for i in range(2,len(ncnames)):
for i in range(6):#,3,4,5,6,7,12]:

  #ax=axarr
  print ax
  print i,iii
  f = Dataset(args.path+'/'+ncfiles[i])
  aa=f.variables[ncnames[i]]
  lmax=f.variables['lmax']
  #print aa[:]
  bb=aa[:]  
  dp = p0/10. / g
  #for i in range(140):
  #  print cc[i,0],i
  plotx=[[]]
  cc = np.zeros_like(bb)
  #for j in range(10):
    #print j
    #cc[j,:]=cumsum_inv(bb[j,:]) * dp 
  #  plotx = plotx + [ax.plot(lmax,cc[j,:], label=j)]
  cc_intp = cc_plot[i]#np.sum(cc,axis=0)
  plot = ax.plot(lmax,cc_intp,linestyles[i],linewidth=linewidths[i],color=colors[i],label=labels[i])
  ax.set_xscale('log')
  ax.set_ylabel('l')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  ax.set_title(args.name)
  iii=iii+1
ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg'
plt.savefig(outname+'.pdf',format='pdf',dpi=400)




