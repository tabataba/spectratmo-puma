#!/bin/bash

START=$PWD

DATA="data/augier360/"
PLOTS="plots/augier_eddy360/"

cd $DATA

dir=($(ls -d *_zg))

for d in ${dir[*]}
do
  echo $d
  if [ -d $d ]; then
    cd $START 
    #./plot_spec.py -p $DATA/$d -n $d -o $START/$PLOTS
    ./plot_spec_eddy.py -p $DATA/$d -n $d -o $START/$PLOTS -e $DATA/$d'_eddy'
    cd $DATA
    #exit 1
  fi
done
