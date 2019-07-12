#!/bin/bash
cmdStr='th'
declare -a covNmStrs=(
"uncover"
 "cover1"
  "cover2"
  "uncover cover1 cover2")
PMLabs=('danaLab' 'simLab')
idx_subTest_SLP_strs=('91 102' '1 7')
cmdPre='main.lua -finalPredictions'
#for i in ${covArgs[@]}; do  # will separate by space to generate elements
#echo $i
#done
#for labNm in $PMLabs
for ((j=1; j<${#PMLabs[@]}; j++))
do
for ((i=2; i<${#covNmStrs[@]}; i++))
do
#$cmdStr main.lua -covNmStr "${covNmStrs[$i]}" -PMlab $labNm
#$cmdStr $cmdPre -branch SLP/danaLab/cov-u12 -covNmStr "${covNmStrs[$i]}" -PMlab ${PMLabs[$j]} -idx_subTest_SLP_str "${idx_subTest_SLP_strs[$j]}"
$cmdStr $cmdPre -branch SLP/danaLab/covRGB-u12  -if_SLPRGB -covNmStr "${covNmStrs[$i]}" -PMlab ${PMLabs[$j]} -idx_subTest_SLP_str "${idx_subTest_SLP_strs[$j]}"
#$cmdStr $cmdPre -loadModel ~/exp/pose-hg-train/umich-stacked-hourglass/umich-stacked-hourglass.t7  -covNmStr "${covNmStrs[$i]}" -PMlab ${PMLabs[$j]} -idx_subTest_SLP_str "${idx_subTest_SLP_strs[$j]}"
$cmdStr $cmdPre -loadModel ~/exp/pose-hg-train/umich-stacked-hourglass/umich-stacked-hourglass.t7  -if_SLPRGB -covNmStr "${covNmStrs[$i]}" -PMlab ${PMLabs[$j]} -idx_subTest_SLP_str "${idx_subTest_SLP_strs[$j]}"
done
done