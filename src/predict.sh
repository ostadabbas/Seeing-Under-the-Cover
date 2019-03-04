#!/bin/bash
mdlLs=(
'SYN/RR_P7_15_A00'
'SYN/RR_P7_15_A00_wns'
'SYN/RR_P7_15_A00_gauFt'
'SURREAL/SURREAL_10000'
'SURREAL/SURREAL_10000_wns'
'SURREAL/SURREAL_10000_gauFt'
)

for mdlNm in "${mdlLs[@]}"
do
echo working on ${mdlNm}
if [[ $mdlNm = *"_gauFt" ]]; then
    echo $mdlNm contains _gauFt
    suf1='-ifGaussFt'
else
    suf1=''
fi
# change the command here for different jobs
th main.lua -branch $mdlNm -dataset AC2d -expID ts1_45 -finalPredictions $suf1
done
echo job done!
#th main.lua