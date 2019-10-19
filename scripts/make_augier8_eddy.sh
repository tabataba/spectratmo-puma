#!/bin/bash

START=$PWD

SPECT=~/soft/spectratmo/ #/spectrun.py

cd pumas_yixiong/pumas

PUMA=$PWD
DATA="data/augier360/"


files=($(ls -d *8*zg.nc))

for d in ${files[*]}
do
  echo $d
  if [ -f $d ]; then
    NAME=${d:: -3}_eddy
    echo $d
    echo $NAME
    cd $SPECT
    ./spectrun.py -n $NAME -p $PUMA/$d -o $START/$DATA -e 1 -i 240
    cd $PUMA
    #exit 1
  fi
done
