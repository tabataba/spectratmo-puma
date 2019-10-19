#!/bin/bash

START=$PWD

SPECT=~/soft/spectratmo/ #/spectrun.py

#cd pumas_yixiong/pumas
#PUMA=$PWD
#DATA="data/augier360/"

DATA="data/augier_pumagt360_ex2/"


#for inmu in 0 1
#do

cd /network/group/aopp/planetary/PLR005_TABATABAVAKILI_PUMAGT/pumagt_arcb_parameters

PUMA=$PWD

dir=($(ls -d *)) # all
dir=($(ls -d  rev53_r0.0625_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus10.00_tausurf360_nmu*)) #specific

counter=0

for d in ${dir[*]}
do
  echo $d
  if [ -d $d ]; then
    cd $d
    pos=$(echo `expr index "$d" p`)
    ps=${d:$pos+4:3}
    pos2=$(echo `expr index "$d" psurf`)
    ps2=${d:$pos+4:4}
    #pos3=$(echo `expr index "$d" "pref"`)
    #let "pos4=pos+4-pos3"
    #echo $pos,$pos4,$pos3
    #ps3=${d:$pos+4:}
    #echo $ps $ps2 $ps3
    #f="PUMAG.010"
    echo $ps $ps2
    if [ $ps2 == 0.02 ] || [ $ps2 == 30.0 ]; then ps=$ps2; fi
    echo $ps 
    LASTYEAR=0
    for YEAR in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    do
      f="PUMAG."$(printf %03d $YEAR)
      fnc="PUMAG."$(printf %03d $YEAR)".nc"
      if [ -f $f ]; then
        LASTYEAR=$YEAR
      fi
      echo $LASTYEAR

    done

      YEAR=$LASTYEAR
      f="PUMAG."$(printf %03d $YEAR)
      fnc="PUMAG."$(printf %03d $YEAR)".nc"
      echo $f
      echo $fnc
      if [ -f $f ]; then #&& [ ! -f $START/data/lorenz_en/$d/$f".npy" ]; then
        if [ ! -f $fnc ]; then  #|| [ $ps != 1.0 ]; then
          echo $f
          echo $ps
          /network/group/aopp/planetary/PLR005_TABATABAVAKILI_PUMAGT/pp_psurf/burn7.x $f $fnc <$START/namelist_ps$ps.nl
        fi
          cd $START
        
        let "counter+=1"
    NAME=$d #${d:: -3}
    NAMEE=$d"_eddy" #${d:: -3}_eddy
    echo $d
    echo $NAME
    cd $SPECT
    echo $counter spectrun
    ./spectrun.py -n $NAME -p $PUMA/$d/$fnc -o $START/$DATA -e 0 -i 30
    #cp gamma  to eddy
    mkdir $START/$DATA/$NAMEE
    cp $START/$DATA/$NAME/gamma.nc $START/$DATA/$NAMEE/.
    echo $counter spectrunE 
    ./spectrun.py -n $NAMEE -p $PUMA/$d/$fnc -o $START/$DATA -e 1 -i 30
     
    #cd $PUMA
    cd $START/$DATA/$NAME
    echo $YEAR > year.dat
    cd $START/$DATA/$NAMEE
    echo $YEAR > year.dat

          #if [ ! -d $DATA/$d ]; then mkdir $DATA/$d; fi
          #echo -p $PUMA/$d/$fnc
          #echo -o $START/data/lorenz_en/$d/$f
          #./puma_energy_boer3.py -p $PUMA/$d/$fnc -o $START/$DATA/$d/$f
          #cd $PUMA
          #cd $d
          #if [ $YEAR -ne "010" ]; then rm $fnc; fi
          #exit 1
        #fi
      fi
    #done
    #exit 1
    cd $PUMA
  fi
done
cd $START 


exit 0




#NAME1=rev53_r0.125_res64_radius1.00_taufr1.0_psurf1.0_pref1.0_taus2.00_tausurf360_nmu$inmu
#cd $NAME1
#PUMA=$PWD
#DATA="data/augier_new360_pumagt_360/"

#files=($(ls -d *zg.nc))
#files=($(ls -d PUMAG.025.nc))

#ls

#for d in ${files[*]}
#do
#  echo $d
#  if [ -f $d ]; then
#    NAME=$NAME1 #${d:: -3}
#    NAMEE=$NAME1"_eddy" #${d:: -3}_eddy
#    echo $d
#    echo $NAME
#    cd $SPECT
#    ./spectrun.py -n $NAME -p $PUMA/$d -o $START/$DATA -e 0 -i 360
#    #cp gamma  to eddy
#    mkdir $START/$DATA/$NAMEE
#    cp $START/$DATA/$NAME/gamma.nc $START/$DATA/$NAMEE/.
#    ./spectrun.py -n $NAMEE -p $PUMA/$d -o $START/$DATA -e 1 -i 360
#    cd $PUMA
    #exit 1
#  fi
#done

#done

