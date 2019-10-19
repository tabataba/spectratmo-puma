#!/bin/bash

START=$PWD

SPECT=~/soft/spectratmo/ #/spectrun.py

#cd pumas_yixiong/pumas
#PUMA=$PWD
#DATA="data/augier360/"


for inmu in 0 1
do

cd /network/group/aopp/planetary/PLR005_TABATABAVAKILI_PUMAGT/

#NAME1=rev53_r0.125_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus2.00_tausurf360_nmu$inmu
#cd $NAME1
PUMA=$PWD
DATA="data/augier_pumag_yixiong3/"

mkdir $DATA

#files=($(ls -d *zg.nc))
files=($(ls -d gh*.nc))
files=($(ls -d pumas_yixiong/pumas_00*_zg.nc))
files=($(ls -d pumag_yixiong3/nc2/*_zg.nc))
#ls

for d in ${files[*]}
do
  echo $d
  if [ -f $d ]; then
    NAME=$d #${d:: -3}
    NAMEE=$d"_eddy" #${d:: -3}_eddy
    echo $d
    echo $NAME
    cd $SPECT
    ./spectrun.py -n $NAME -p $PUMA/$d -o $START/$DATA -e 0 -i 30
    #cp gamma  to eddy
    mkdir $START/$DATA/$NAMEE
    cp $START/$DATA/$NAME/gamma.nc $START/$DATA/$NAMEE/.
    ./spectrun.py -n $NAMEE -p $PUMA/$d -o $START/$DATA -e 1 -i 30
    cd $PUMA
    #exit 1
  fi
done

done

