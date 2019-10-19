#!/bin/bash

START=$PWD

DATA="data/augier360/"
#DATA="data/augier_pumagt30_alpha/"

DATA="data/augier_pumas30_lowfric/"

#DATA="data/augier_pumagt_360"
#DATA="data/augier_pumagt360_ex2"
#DATA="data/augier_pumagt360_omega0125_alpha"
#DATA="data/augier_pumagt360_seasonal2/"
#PLOTS="plots/augier_pumagt_eddy360/"

PLOTS="plots/augier_pumas30_lowfric"

#PLOTS="plots/augier_pumagt360_seasonal2"
#PLOTS="plots/augier_pumagt30_alpha"

mkdir -p $PLOTS"_0"
mkdir -p $PLOTS"_15"
mkdir -p $PLOTS"_17"

cd $DATA

pwd
ls
#dir=($(ls -d rev*_nmu[0-1]))
#dir=($(ls -d *_zg))
dir=($(ls -d *))


for d in ${dir[*]}
do
  echo $d
  if [ -d $d ]; then
    cd $START 
    #./plot_spec.py -p $DATA/$d -n $d -o $START/$PLOTS
    ./plot_spec_eddy2.py -p $DATA/$d -n $d -o $START/$PLOTS"_0" -e $DATA/$d'_eddy' -l 0
    ./plot_spec_eddy2.py -p $DATA/$d -n $d -o $START/$PLOTS"_15" -e $DATA/$d'_eddy' -l 15
    ./plot_spec_eddy2.py -p $DATA/$d -n $d -o $START/$PLOTS"_17" -e $DATA/$d'_eddy' -l 17
    cd $DATA
    #exit 1
  fi
done

cd $START
cp -rf $PLOTS ~/Dropbox/DPhil/thesis/.
