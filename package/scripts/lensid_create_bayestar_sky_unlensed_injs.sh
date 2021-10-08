#!/bin/sh

start=0
n=1004
odir='check'

Help(){
   # Display Help
    echo "Script to create bayestar skymaps or unlensed injections"
    echo
    echo "options:"
    echo "h     Print this Help."
    echo "o     output directory path"
    echo "s     start index"
    echo "n     no. of events"
    echo "i     .npz unlensed injection parameters file "
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
        p) psd_file=${OPTARG};;
        h) # display Help
            Help
            exit;;
    esac
done

out=$odir'/unlensed'
echo $out
export OMP_NUM_THREADS=8

for index in $(seq $start 1 $(($start + $n - 1))) #1004 
do
    path= echo ${out}'/'${index}'/'
    FILE=${out}'/'${index}'/'0.fits
    #if  [ ! -f "$FILE" ] 
    #then

    lensid_create_unlensed_inj_xmls --index ${index} --odir ${out} \
    --infile ${infile}

    scp ${psd_file} ${out}'/'psd.xml


    bayestar-realize-coincs \
    -o ${out}'/'${index}'/'coinc.xml \
    ${out}'/'${index}'/'inj.xml --reference-psd ${out}'/'psd.xml \
    --detector H1 L1 V1 \
    --measurement-error gaussian-noise \
    --net-snr-threshold 2.0 \
    --min-triggers 1 \
    --snr-threshold 1 \
    -P

    bayestar-localize-coincs ${out}'/'${index}'/'coinc.xml -o ${out}'/'${index}'/'

    ligo-skymap-plot ${out}'/'${index}'/'0.fits -o ${out}'/'${index}'_'skymap.png \
    --annotate --contour 50 90

    #fi
    lensid_sky_injs_cart  -index ${index} -lensed n -odir ${odir} -indir ${odir} \
    -infile ${infile}

done