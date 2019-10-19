#!/bin/bash

START=$PWD

#DATA="data/augier360/"
#DATA="data/augier_pumagt30_alpha/"

#DATA="data/augier_pumagt_360"
#DATA="data/augier_pumag"
DATA="data/augier_pumag_low/"
DATA="data/augier_pumag_yixiong3/pumag_yixiong3/nc2/"
#PLOTS="plots/augier_pumagt_eddy360/"

#PLOTS="plots/new_augier360"
PLOTS="plots/augier_pumag_low"
PLOTS="plots/augier_pumag_yixiong3/"
#PLOTS="plots/augier_pumagt30_alpha"

mkdir $PLOTS

cd $DATA

#dir=($(ls -d rev*_nmu[0-1]))
#dir=($(ls -d gh*.nc))
dir=($(ls -d *zg.nc))

for d in ${dir[*]}
do
  echo $d
  if [ -d $d ]; then
    cd $START 
    #./plot_spec.py -p $DATA/$d -n $d -o $START/$PLOTS
    ./plot_spec_eddy2.py -p $DATA/$d -n $d -o $START/$PLOTS -e $DATA/$d'_eddy'
    cd $DATA
    #exit 1
  fi
done
