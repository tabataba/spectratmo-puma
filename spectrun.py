#!/network/home/aopp/tabatabavakili/soft/spectatmosve/bin/python

import sys
import os
from spectratmo.datasets.puma import PUMADataSet

from argparse import ArgumentParser
parser = ArgumentParser()

if __name__ == "__main__":

  parser.add_argument("-n","--name",action="store",type=str)
  parser.add_argument("-p","--path",action="store",type=str)
  parser.add_argument("-o","--output",action="store",type=str)  
  parser.add_argument("-e","--eddy",action="store",type=int)
  parser.add_argument("-i","--instants",action="store",type=int)

  args, unknown = parser.parse_known_args()

  path=args.path
  if args.path==None or args.output==None or args.name==None:
    print "define args!"
    sys.exit(1)

  if args.eddy==1:
    print 'eddies!'
    make_eddy=1
  else:
    make_eddy=0

  print make_eddy

  #sys.exit()

  configpath = args.output+'/'+args.name+'/'
  config=args.output+'/'+args.name+'/'+'config.py'

  if args.instants==None:
    instants=30
  else:
    instants=args.instants
  if not os.path.exists(configpath):
    os.makedirs(configpath)
  f = open(config, 'w')
  f.write("name = '"+args.name+"'"+"\n")
  f.write("instants = range("+str(instants)+")")
  f.close()

  ds = PUMADataSet(name=args.name,path_file=args.path,puma_path_result_base=args.output,make_eddy=make_eddy)
  ds.spectra.compute_tmean()
