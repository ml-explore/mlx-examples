#!/bin/bash

for k in `seq 500`
do

./evo_lama.sh > results_evo_lama_${k}_${RANDOM}_${RANDOM}.loglama

done
