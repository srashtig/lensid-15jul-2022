#!/bin/sh

start=0
#n=2813 
n=300
odir='check'
Help(){
   # Display Help
    echo "Script to create bayestar skymaps or lensed injections"
    echo
    echo "options:"
    echo "h     Print this Help."
    echo "o     output directory path"
    echo "s     start index"
    echo "n     no. of events"
    echo "i     .npz lensed injection parameters file "
    echo "p     input psd XML file"
    echo
}
while getopts o:s:n:i:p:h flag
do
    case "${flag}" in
        o) odir=${OPTARG};;
        s) start=${OPTARG};;
        n) n=${OPTARG};;
        i) infile=${OPTARG};;
        p) psdfile=${OPTARG};;
        h) # display Help
            Help
            exit;;
    esac
done

out=$odir'/lensed'
echo ${psdfile}
export OMP_NUM_THREADS=8
#2813
for index in $(seq $start 1 $(($start + $n - 1))) #300
do
for img in $(seq 0 1)
do
path= echo ${out}'/'${index}'/'${img}'/'
FILE=${out}'/'${index}'/'${img}'/'0.fits
#if  [ ! -f "$FILE" ] 
#then
lensid_create_lensed_inj_xmls --index ${index} --img ${img} --odir ${out} --infile ${infile}

#bayestar-sample-model-psd -o ${out}'/'${index}'/'psd.xml --H1=aLIGOZeroDetHighPower --L1=aLIGOZeroDetHighPower --V1=AdvVirgo

scp ${psdfile} ${out}/'psd.xml'

bayestar-realize-coincs \
-o ${out}'/'${index}'/'${img}'/'coinc.xml \
${out}'/'${index}'/'${img}'/'inj.xml --reference-psd ${out}/'psd.xml' \
--detector H1 L1 V1 \
--measurement-error gaussian-noise \
--net-snr-threshold 2.0 \
--min-triggers 1 \
--snr-threshold 1 \
-P


# change/check snr thresholds, net-snr-threshold 6.0
bayestar-localize-coincs ${out}'/'${index}'/'${img}'/'coinc.xml -o ${out}'/'${index}'/'${img}'/'

ligo-skymap-plot ${out}'/'${index}'/'${img}'/'0.fits -o ${out}'/'${index}'_'${img}'_'skymap.png --annotate --contour 50 90

#fi
done
lensid_sky_injs_cart  -index ${index} -lensed y -odir ${odir} -indir ${odir} -infile ${infile}
done
