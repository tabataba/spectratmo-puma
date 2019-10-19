#!/bin/bash

START=$PWD

SPECT=~/soft/spectratmo/ #/spectrun.py

#cd pumas_yixiong/pumas
#PUMA=$PWD
#DATA="data/augier360/"

DATA="data/augier_pumagt360_seasonal2/"

mkdir -p $DATA

#for inmu in 3.6 36 360
for NAME1 in rev53_r0.125_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus0.00_tausurf3.6_nmu1 rev53_r1.0_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus0.00_tausurf3.6_nmu1 rev53_r0.125_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus10.00_tausurf360_nmu1 rev53_r1.0_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus10.00_tausurf360_nmu1
do

cd /network/group/aopp/planetary/PLR005_TABATABAVAKILI_PUMAGT/pumagt_arcb_parameters


#NAME1="rev53_r0.125_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus0.00_tausurf"$inmu"_nmu1"


cd $NAME1
ls 
PUMA=$PWD
#DATA="data/augier_pumagt360_omega0125_alpha/"

#files=($(ls -d *zg.nc))
#files=($(ls -d PUMAG.018.nc))
files=($(ls -d PUMAG_NWPD12_M.001))
#PUMAG_NWPD12_M.001

#ls
#cd data
#mkdir $DATA

for d in ${files[*]}
do
  echo $d
  if [ -f $d ]; then
    NAME=$NAME1 #${d:: -3}
    NAMEE=$NAME1"_eddy" #${d:: -3}_eddy
    echo $d
    echo $NAME
      f=$d
      fnc=$d".nc"
      #echo $f
      if [ -f $f ]; then
        if [ ! -f $fnc ]; then  #|| [ $ps != 1.0 ]; then
          echo $f
          echo $ps
          /network/group/aopp/planetary/PLR005_TABATABAVAKILI_PUMAGT/pp_psurf/burn7.x $f $fnc <$START/namelist_ps1.0.nl
        fi
    cd $SPECT
    ./spectrun.py -n $NAME -p $PUMA/$fnc -o $START/$DATA -e 0 -i 360
    #cp gamma  to eddy
    mkdir $START/$DATA/$NAMEE
    cp $START/$DATA/$NAME/gamma.nc $START/$DATA/$NAMEE/.
    ./spectrun.py -n $NAMEE -p $PUMA/$fnc -o $START/$DATA -e 1 -i 360
    cd $PUMA
    #exit 1
      fi
  fi
done

done

