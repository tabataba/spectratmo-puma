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
parser.add_argument("-l","--legend",action="store",type=int)  

args, unknown = parser.parse_known_args()

size=7  #10

xsize=1.4*size
ysize=0.8*size

if args.legend < 10: 
  lfont=1
  ileg=0
else:
  lfont=args.legend
  ileg=1

plt.rc('legend',**{'fontsize':lfont})

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
#mmax=f.variables['mmax']
lsize=len(lmax[:])

print 'normal'

fig, axarr = plt.subplots(4, 4, sharex='row', figsize=(3*10,4*10))

axarr = axarr.ravel()

cc_sav=np.empty((len(ncnames),lsize))
cc_sav2=np.empty((len(ncnames),10,lsize))
#aa_sav=np.empty((len(ncnames),10,lsize))

for i in range(0,len(ncnames)):
  ax=axarr[i]
  print ax
  f = Dataset(args.path+'/'+ncfiles[i])
  aa=f.variables[ncnames[i]]
  lmax=f.variables['lmax']
  #print aa[:]
  #aa_sav=aa[:]
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
    if i == 0 or i == 1: # or i == 2:   #add 2 to have C be shown as tendency
      cc[j,:]= bb[j,:] * dp
    else:
      cc[j,:]= cumsum_inv(bb[j,:]) * dp
    plotx = plotx + [ax.plot(lmax,cc[j,:], label=j)]
    cc_sav2[i,j,:]=cc[j,:]
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
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(ncnames[i])

plt.tight_layout()
outname=args.output+'/'+args.name
plt.savefig(outname+'.pdf',format='pdf',dpi=200)


#colors=['k','DarkOrange','b','g','DarkOrange','DarkOrange']
#linestyles=['-','-','-','-.',':','--']
#linewidths=[4,2,2,2,1.5,1.5]
#labels=[r'$\Pi$',r'$\Pi_K$',r'$\Pi_A$',r'$\mathcal{C}$',r'$\Pi_{Krot}$',r'$\Pi_{Kdiv}$']



if not args.eddypath==None:
  print 'eddy'
  
  fig, axarr = plt.subplots(4, 4, sharex='row', figsize=(3*10,4*10))

  axarr = axarr.ravel()

  cce_sav=np.empty((len(ncnames),lsize))
  cce_sav2=np.empty((len(ncnames),10,lsize))
  aae_sav=np.empty((len(ncnames),lsize))

  for i in range(0,len(ncnames)):
    ax=axarr[i]
    print ax
    print args.eddypath+'/'+ncfiles[i]
    fe = Dataset(args.eddypath+'/'+ncfiles[i])
    aae=fe.variables[ncnames[i]]
    aaw_sav=aae[:]
    lmax=fe.variables['lmax']
    #print aa[:]
    bbe=aae[:]
    #dp = p0/10. / g
    #for i in range(140):
    #  print cc[i,0],i
    plotx=[[]]
    cce = np.zeros_like(bbe)
    if i == 8 or i==9 or i==10:
      dp = 1
    else:
      dp = p0/10. / g
    for j in range(10):
      if i == 0 or i == 1: #  or i == 2:
        cce[j,:]= bbe[j,:] * dp
      else:
        cce[j,:]= cumsum_inv(bbe[j,:]) * dp
      plotx = plotx + [ax.plot(lmax,cce[j,:], label=j)]
      cce_sav2[i,j,:]=cce[j,:]
    if i==0: ax.legend()
    #if i==13:
    #  cc_intp = np.mean(cc,axis=0)
    #else:
    cce_intp = np.sum(cce,axis=0)
    cce_sav[i,:]=cce_intp[:]
    if i == 0 or i == 1:
      plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-5.))),linewidth=3.0,color='k')
      plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-5./3.))),linewidth=3.0,color='k')
      plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-3.))),linewidth=3.0,color='k')
    #print(lmax[1::])
    #print(1E7*(pow(lmax[1::],(-5./3.))))
    plot = ax.plot(lmax,cce_intp,linewidth=4.0,color='b')
    ax.set_xscale('log')
    if i < 2: ax.set_yscale('log')
    #if i == 8 or i==9 or i==10: 
    ax.set_xlabel('wavenumber')
    ax.set_ylabel(ncnames[i])

  plt.tight_layout()
  outname=args.output+'/'+args.name+'_eddy'
  plt.savefig(outname+'.pdf',format='pdf',dpi=200)

  print 'zonal'

  fig, axarr = plt.subplots(4, 4, sharex='row', figsize=(3*10,4*10))

  axarr = axarr.ravel()

  ccz_sav=np.empty((len(ncnames),lsize))
  ccz_sav2=np.empty((len(ncnames),10,lsize))

  for i in range(0,len(ncnames)):
    ax=axarr[i]
    print ax
    #fz = Dataset(args.eddypath+'/'+ncfiles[i])
#    aaz=aa_sav[i,:]-aae_sav[i,:]#fe.variables[ncnames[i]]
    lmax=f.variables['lmax']
    #print aa[:]
#    bbz=aaz[:]
    #dp = p0/10. / g
    #for i in range(140):
    #  print cc[i,0],i
    plotx=[[]]
#    ccz = cc_sav - cc
#    ccz = np.zeros_like(bbz)
#    if i == 8 or i==9 or i==10:
#      dp = 1
#    else:
#      dp = p0/10. / g
#    for j in range(10):
#      if i == 0 or i == 1:
#        ccz[j,:]= bbz[j,:] * dp
#      else:
#        ccz[j,:]= cumsum_inv(bbz[j,:]) * dp
#      plotx = plotx + [ax.plot(lmax,ccz[j,:], label=j)]
#    if i==0: ax.legend()
    #if i==13:
    #  cc_intp = np.mean(cc,axis=0)
    #else:
#    ccz_intp = np.sum(ccz,axis=0)
#    ccz_sav[i,:]=ccz_intp[:]
    ccz_intp=cc_sav[i,:]-cce_sav[i,:]
    ccz_sav2[i,:,:]=cc_sav2[i,:,:]-cce_sav2[i,:,:]
    ccz_sav[i,:]=ccz_intp[:]
    if i == 0 or i == 1:
      plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-5.))),linewidth=3.0,color='k')
      plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-5./3.))),linewidth=3.0,color='k')
      plot = ax.plot(lmax[1::],1E7*(pow(lmax[1::],(-3.))),linewidth=3.0,color='k')
    #print(lmax[1::])
    #print(1E7*(pow(lmax[1::],(-5./3.))))
    plot = ax.plot(lmax,ccz_intp,linewidth=4.0,color='b')
    #plot = ax.plot(lmax,cc_sav[i,:]-cce_sav[i,:],linewidth=4.0,color='r')
    ax.set_xscale('log')
    if i < 2: ax.set_yscale('log')
    #if i == 8 or i==9 or i==10: 
    ax.set_xlabel('wavenumber')
    ax.set_ylabel(ncnames[i])

  plt.tight_layout()
  outname=args.output+'/'+args.name+'_zonal'
  plt.savefig(outname+'.pdf',format='pdf',dpi=200)


matplotlib.rcParams.update({'font.size': 17})




cc_p=np.empty((10,10,lsize))
cce_p=np.empty((10,10,lsize))
ccz_p=np.empty((10,10,lsize))

#C
cc_p[3,:,:]  = cc_sav2[2,:,:]
cce_p[3,:,:] = cce_sav2[2,:,:]
ccz_p[3,:,:] = ccz_sav2[2,:,:]
#Pi_A
cc_p[2,:,:]  = cc_sav2[3,:,:] + cc_sav2[4,:,:]
cce_p[2,:,:] = cce_sav2[3,:,:] + cce_sav2[4,:,:]
ccz_p[2,:,:] = ccz_sav2[3,:,:] + ccz_sav2[4,:,:]
#Pi_K
cc_p[1,:,:]  = cc_sav2[5,:,:] + cc_sav2[6,:,:]
cce_p[1,:,:] = cce_sav2[5,:,:] + cce_sav2[6,:,:]
ccz_p[1,:,:] = ccz_sav2[5,:,:] + ccz_sav2[6,:,:]
#Pi_tot
cc_p[0,:]  = cc_p[1,:,:] + cc_p[2,:,:]
cce_p[0,:] = cce_p[1,:,:] + cce_p[2,:,:]
ccz_p[0,:] = ccz_p[1,:,:] + ccz_p[2,:,:]

#Pi_Krot
cc_p[4,:,:]  = cc_sav2[7,:,:]
cce_p[4,:,:] = cce_sav2[7,:,:]
ccz_p[4,:,:] = ccz_sav2[7,:,:]
#Pi_Kdiv
cc_p[5,:,:]  = cc_p[1,:,:] - cc_p[4,:,:]
cce_p[5,:,:] = cce_p[1,:,:] - cce_p[4,:,:]
ccz_p[5,:,:] = ccz_p[1,:,:] - ccz_p[4,:,:]

#EK
cc_p[6,:,:]  = cc_sav2[1,:,:]
cce_p[6,:,:] = cce_sav2[1,:,:]
ccz_p[6,:,:] = ccz_sav2[1,:,:]

#EA

cc_p[7,:,:]  = cc_sav2[0,:,:]
cce_p[7,:,:] = cce_sav2[0,:,:]
ccz_p[7,:,:] = ccz_sav2[0,:,:]

#F_A
cc_p[8,:,:]  = cc_sav2[8,:,:]
cce_p[8,:,:] = cce_sav2[8,:,:]
ccz_p[8,:,:] = ccz_sav2[8,:,:]

#F_K
cc_p[9,:,:]  = cc_sav2[9,:,:]  + cc_sav2[10,:,:]
cce_p[9,:,:] = cce_sav2[9,:,:] + cce_sav2[10,:,:]
ccz_p[9,:,:] = ccz_sav2[9,:,:] + ccz_sav2[10,:,:]

































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

cc_plot=np.empty((10,lsize))
cce_plot=np.empty((10,lsize))
ccz_plot=np.empty((10,lsize))

#C
cc_plot[3,:]  = cc_sav[2,:]
cce_plot[3,:] = cce_sav[2,:]
ccz_plot[3,:] = ccz_sav[2,:]
#Pi_A
cc_plot[2,:]  = cc_sav[3,:] + cc_sav[4,:]
cce_plot[2,:] = cce_sav[3,:] + cce_sav[4,:]
ccz_plot[2,:] = ccz_sav[3,:] + ccz_sav[4,:]
#Pi_K
cc_plot[1,:]  = cc_sav[5,:] + cc_sav[6,:]
cce_plot[1,:] = cce_sav[5,:] + cce_sav[6,:]
ccz_plot[1,:] = ccz_sav[5,:] + ccz_sav[6,:]
#Pi_tot
cc_plot[0,:]  = cc_plot[1,:] + cc_plot[2,:]
cce_plot[0,:] = cce_plot[1,:] + cce_plot[2,:]
ccz_plot[0,:] = ccz_plot[1,:] + ccz_plot[2,:]

#Pi_Krot
cc_plot[4,:]  = cc_sav[7,:]
cce_plot[4,:] = cce_sav[7,:]
ccz_plot[4,:] = ccz_sav[7,:]
#Pi_Kdiv
cc_plot[5,:]  = cc_plot[1,:] - cc_plot[4,:]
cce_plot[5,:] = cce_plot[1,:] - cce_plot[4,:]
ccz_plot[5,:] = ccz_plot[1,:] - ccz_plot[4,:]

#EK
cc_plot[6,:]  = cc_sav[1,:]
cce_plot[6,:] = cce_sav[1,:]
ccz_plot[6,:] = ccz_sav[1,:]

#EA

cc_plot[7,:]  = cc_sav[0,:]
cce_plot[7,:] = cce_sav[0,:]
ccz_plot[7,:] = ccz_sav[0,:]

#F_A
cc_plot[8,:]  = cc_sav[8,:]
cce_plot[8,:] = cce_sav[8,:]
ccz_plot[8,:] = ccz_sav[8,:]

#F_K
cc_plot[9,:]  = cc_sav[9,:]  + cc_sav[10,:]
cce_plot[9,:] = cce_sav[9,:] + cce_sav[10,:]
ccz_plot[9,:] = ccz_sav[9,:] + ccz_sav[10,:]


np.savetxt(outname+'.txt',cce_plot[6,:])


colors=['DarkOrange','b']
linestyles=['-','-']
linewidths=[4,4]
labels=[r'$E_K$',r'$E_A$']

ecolors=['DarkOrange','b']
elinestyles=['-.','-.']
elinewidths=[2,2]
elabels=[r'$E_{K,eddy}$',r'$E_{A,eddy}$']

zcolors=['DarkOrange','b']
zlinestyles=['--','--']
zlinewidths=[2,2]
zlabels=[r'$E_{K,zonal}$',r'$E_{A,zonal}$']

fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
#for i in range(2,len(ncnames)):
for i in [6,7]:

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
  plot = ax.plot(lmax,cc_intp,linestyles[iii],linewidth=linewidths[iii],color=colors[iii],label=labels[iii])
  plot = ax.plot(lmax,cce_plot[i],elinestyles[iii],linewidth=elinewidths[iii],color=ecolors[iii],label=elabels[iii])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[iii],linewidth=zlinewidths[iii],color=zcolors[iii],label=zlabels[iii])

  plot = ax.plot(lmax[5::],1E7*(pow(lmax[5::],(-5.))),linewidth=1.0,color='k')
  plot = ax.plot(lmax[5::],1E7*(pow(lmax[5::],(-5./3.))),linewidth=1.0,color='k')
  plot = ax.plot(lmax[5::],1E7*(pow(lmax[5::],(-3.))),linewidth=1.0,color='k')

  x1=lmax[-5]
  y1=1E7*(pow(lmax[-5],(-5.)))
  y2=1E7*(pow(lmax[-5],(-5./3.)))
  y3=1E7*(pow(lmax[-5],(-3.)))
  ax.annotate(r'$k^{-5}$',xy=(x1,y1),xytext=(x1*0.6,y1*1.0))
  ax.annotate(r'$k^{-5/3}$',xy=(x1,y2),xytext=(x1*0.8,y2*1.8))
  ax.annotate(r'$k^{-3}$',xy=(x1,y3),xytext=(x1*0.8,y3*3))

  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$E[l]$ (J/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+'1en_'+args.name+'_tg_ez'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)






















colors=['k','DarkOrange','b','g','DarkOrange','DarkOrange']
linestyles=['-','-','-','-.',':','--']
linewidths=2*[4,2,2,2,1.5,1.5]
labels=[r'$\Pi$',r'$\Pi_K$',r'$\Pi_A$',r'$\mathcal{C}$',r'$\Pi_{Krot}$',r'$\Pi_{Kdiv}$']




fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

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
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)




colors=['k','r','b','g','DarkOrange','purple']
linestyles=['-','-','-','-','-','-']
linewidths=[4,3,3,3,2,2]
labels=[r'$\Pi$',r'$\Pi_K$',r'$\Pi_A$',r'$\mathcal{C}$',r'$\Pi_{Krot}$',r'$\Pi_{Kdiv}$']

ecolors=['k','r','b','g','DarkOrange','purple']
elinestyles=['-.','-.','-.','-.','-.','-.']
elinewidths=linewidths#3*[4,2,2,2,1.5,1.5]
elabels=[r'$\Pi_{eddy}$',r'$\Pi_{K,eddy}$',r'$\Pi_{A,eddy}$',r'$\mathcal{C}_{eddy}$',r'$\Pi_{Krot,eddy}$',r'$\Pi_{Kdiv,eddy}$']

zcolors=['k','r','b','g','DarkOrange','purple']
zlinestyles=['--','--','--','--','--','--']
zlinewidths=linewidths#3*[4,2,2,2,1.5,1.5]
zlabels=[r'$\Pi_{zonal}$',r'$\Pi_{K,zonal}$',r'$\Pi_{A,zonal}$',r'$\mathcal{C}_{zonal}$',r'$\Pi_{Krot,zonal}$',r'$\Pi_{Kdiv,zonal}$']

fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

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
  plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)





fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
#for i in range(2,len(ncnames)):
for i in range(3):#,3,4,5,6,7,12]:

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
  plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_half1'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)





fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
#for i in range(2,len(ncnames)):
for i in range(3,6):#,3,4,5,6,7,12]:

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
  plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_half2'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)





fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

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
  #plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  #plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_onlytotal'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)





fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

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
  #plot = ax.plot(lmax,cc_intp,linestyles[i],linewidth=linewidths[i],color=colors[i],label=labels[i])
  plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  #plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_onlyeddy'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)

fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

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
  #plot = ax.plot(lmax,cc_intp,linestyles[i],linewidth=linewidths[i],color=colors[i],label=labels[i])
  #plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_onlyzonal'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)

fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
#for i in range(2,len(ncnames)):
for i in range(4):#,3,4,5,6,7,12]:

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
  plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_noKrotdiv'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)



fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
#for i in range(2,len(ncnames)):
for i in [1,4,5]:#,3,4,5,6,7,12]:

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
  plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_onlyK'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)




kmint=np.min(cc_plot[1])
kmine=np.min(cce_plot[1])
kminz=np.min(ccz_plot[1])
kmin=np.min([kmint,kmine,kminz])*1.2
kmaxt=np.max(cc_plot[1])
kmaxe=np.max(cce_plot[1])
kmaxz=np.max(ccz_plot[1])
kmax=np.max([kmaxt,kmaxe,kmaxz])*1.2

fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
#for i in range(2,len(ncnames)):
for i in [1]:#,3,4,5,6,7,12]:

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
  plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  #ax.set_ylim([kmin,kmax])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  #ax.set_title(args.name)
  iii=iii+1
if ileg==1: ax.legend()
#plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_onlyKA'
plt.savefig(outname+'.pdf',format='pdf',dpi=200)




colors=['k','DarkOrange','b','g','DarkOrange','DarkOrange','k','k']
linestyles=['-','-','-','-.',':','--','-','-']
linewidths=2*[4,2,2,2,1.5,1.5,1,1]
labels=[r'$\Pi$',r'$\Pi_K$',r'$\Pi_A$',r'$\mathcal{C}$',r'$\Pi_{Krot}$',r'$\Pi_{Kdiv}$','x','x']

fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

x1=5
x2=7

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
for i in [0,1,2,3]:#,3,4,5,6,7,12]:

  print ax
  print i,iii
  f = Dataset(args.path+'/'+ncfiles[i])
  aa=f.variables[ncnames[i]]
  lmax=f.variables['lmax']
  bb=aa[:]
  dp = p0/10. / g
  plotx=[[]]
  cc = np.zeros_like(bb)
  cc_intp = np.mean(cc_p[i,x1:x2,:],axis=0)#np.sum(cc,axis=0)
  plot = ax.plot(lmax,cc_intp,linestyles[i],linewidth=linewidths[i],color=colors[i],label=labels[i])
  #plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  #plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  ax.set_title(r'$p_b=450$ hPa, $p_t=250$ hPa')
  iii=iii+1

#cc_intp = np.mean(cc_p[i,x1:x2,:],axis=1)#np.sum(cc,axis=0)
plot = ax.plot(lmax,cc_p[i,x1,:],'--',linewidth=2,color='purple',label=r'$\mathcal{F}_{\uparrow}[l](p_b)$')
plot = ax.plot(lmax,cc_p[i,x2,:],'-.',linewidth=2,color='purple',label=r'$\mathcal{F}_{\uparrow}[l](p_t)$')
plot = ax.plot(lmax,cc_p[i,x1,:]-cc_p[i,x2,:],'-',linewidth=2,color='purple',label=r'$\Delta^{p_b}_{p_t}\mathcal{F}_{\uparrow}[l]$')

if ileg==1: ax.legend()

plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_F_p'+str(x1)+'-'+str(x2)
plt.savefig(outname+'.pdf',format='pdf',dpi=200)



fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

x1=7
x2=9

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
for i in [0,1,2,3]:#,3,4,5,6,7,12]:

  print ax
  print i,iii
  f = Dataset(args.path+'/'+ncfiles[i])
  aa=f.variables[ncnames[i]]
  lmax=f.variables['lmax']
  bb=aa[:]
  dp = p0/10. / g
  plotx=[[]]
  cc = np.zeros_like(bb)
  cc_intp = np.mean(cc_p[i,x1:x2,:],axis=0)#np.sum(cc,axis=0)
  plot = ax.plot(lmax,cc_intp,linestyles[i],linewidth=linewidths[i],color=colors[i],label=labels[i])
  #plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  #plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  ax.set_title(r'$p_b=250$ hPa, $p_t=50$ hPa')
  iii=iii+1

#cc_intp = np.mean(cc_p[i,x1:x2,:],axis=1)#np.sum(cc,axis=0)
plot = ax.plot(lmax,cc_p[i,x1,:],'--',linewidth=2,color='purple',label=r'$\mathcal{F}_{\uparrow}[l](p_b)$')
plot = ax.plot(lmax,cc_p[i,x2,:],'-.',linewidth=2,color='purple',label=r'$\mathcal{F}_{\uparrow}[l](p_t)$')
plot = ax.plot(lmax,cc_p[i,x1,:]-cc_p[i,x2,:],'-',linewidth=2,color='purple',label=r'$\Delta^{p_b}_{p_t}\mathcal{F}_{\uparrow}[l]$')

if ileg==1: ax.legend()

plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_F_p'+str(x1)+'-'+str(x2)
plt.savefig(outname+'.pdf',format='pdf',dpi=200)


fig, axarr = plt.subplots(1, 1, sharex='row', figsize=(xsize,ysize))

x1=0
x2=5

iii=0
ax=axarr
plot = ax.plot(lmax[:],0.0*lmax[:],linestyle=':',color='k')
for i in [0,1,2,3]:#,3,4,5,6,7,12]:

  print ax
  print i,iii
  f = Dataset(args.path+'/'+ncfiles[i])
  aa=f.variables[ncnames[i]]
  lmax=f.variables['lmax']
  bb=aa[:]
  dp = p0/10. / g
  plotx=[[]]
  cc = np.zeros_like(bb)
  cc_intp = np.mean(cc_p[i,x1:x2,:],axis=0)#np.sum(cc,axis=0)
  plot = ax.plot(lmax,cc_intp,linestyles[i],linewidth=linewidths[i],color=colors[i],label=labels[i])
  #plot = ax.plot(lmax,cce_plot[i],elinestyles[i],linewidth=elinewidths[i],color=ecolors[i],label=elabels[i])
  #plot = ax.plot(lmax,ccz_plot[i],zlinestyles[i],linewidth=zlinewidths[i],color=zcolors[i],label=zlabels[i])
  ax.set_xscale('log')
  ax.set_xlabel('wavenumber')
  ax.set_ylabel(r'$\Pi[l]$ (W/m$^2$)')
  ax.set_title(r'$p_b=950$ hPa, $p_t=450$ hPa')
  iii=iii+1

#cc_intp = np.mean(cc_p[i,x1:x2,:],axis=1)#np.sum(cc,axis=0)
plot = ax.plot(lmax,cc_p[i,x1,:],'--',linewidth=2,color='purple',label=r'$\mathcal{F}_{\uparrow}[l](p_b)$')
plot = ax.plot(lmax,cc_p[i,x2,:],'-.',linewidth=2,color='purple',label=r'$\mathcal{F}_{\uparrow}[l](p_t)$')
plot = ax.plot(lmax,cc_p[i,x1,:]-cc_p[i,x2,:],'-',linewidth=2,color='purple',label=r'$\Delta^{p_b}_{p_t}\mathcal{F}_{\uparrow}[l]$')

if ileg==1: ax.legend()

plt.tight_layout()
outname=args.output+'/'+args.name+'_tg_ez_F_p'+str(x1)+'-'+str(x2)
plt.savefig(outname+'.pdf',format='pdf',dpi=200)
