#!/bin/bash

for algo in SQOPSO
do
echo "ALGORITHM $algo =================================="
(
grep '666.*scale' `ls -ctr results_evo_lama_*`  |grep -v '===>G' |  sed 's/.*666 ===>//g' | sed 's/(.*scale=//g' | awk '{ score[$2, " 0"] += $1 ; num[$2, " 0"] += 9} END { for ( k in num ) { print k, score[k] / num[k], num[k]/9 } }' | awk '{print $1, $2, $3, $4 }' | sed 's/$/     BASELINE/g'
grep "${algo}.*scale" `ls -ctr results_evo_lama_*`  | grep -v '===>G'  | sed 's/^[^:]*://g' | sed 's/ ===>//g' | sed 's/(.*scale=//g' | awk '{ score[$3," ",$1] += $2 ; num[$3," ",$1] += 9} END { for ( k in num ) { print k, score[k] / num[k], num[k]/9 } }'
#grep 'custom.*scale' `ls -ctr results_evo_lama_*`  | sed 's/.* ===>//g' | sed 's/(.*scale=//g' | awk '{ score[$2] += $1 ; num[$2] += 9} END { for ( k in num ) { print k, score[k] / num[k], num[k]/9 } }'
) | sort -n -r -k 2,2 | sed 's/^/"TRAIN,scale=/g' 

echo "IN GENERALIZATION:"
(
grep '666.*scale' `ls -ctr results_evo_lama_*`  | grep '===>G' | sed 's/.*666 ===>G//g' | sed 's/(.*scale=//g' | awk '{ score[$2, " 0"] += $1 ; num[$2, " 0"] += 9} END { for ( k in num ) { print k, score[k] / num[k], num[k]/9 } }' | awk '{print $1, $2, $3, $4 }' | sed 's/$/     BASELINE/g'
grep "${algo}.*scale" `ls -ctr results_evo_lama_*`  |grep '===>G' |  sed 's/^[^:]*://g' | sed 's/ ===>G//g' | sed 's/(.*scale=//g' | awk '{ score[$3," ",$1] += $2 ; num[$3," ",$1] += 9} END { for ( k in num ) { print k, score[k] / num[k], num[k]/9 } }'
#grep 'custom.*scale' `ls -ctr results_evo_lama_*`  | sed 's/.* ===>//g' | sed 's/(.*scale=//g' | awk '{ score[$2] += $1 ; num[$2] += 9} END { for ( k in num ) { print k, score[k] / num[k], num[k]/9 } }'
) | sort -n -r -k 2,2  | sed 's/^/"TEST,scale=/g'
done
