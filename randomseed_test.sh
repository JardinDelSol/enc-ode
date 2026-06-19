#!/bin/bash

for gpu in {4..7}; do
    seed=$((2018 + gpu))
    for datatype in fdg tau amyloid; do
        nohup python main.py -gpu_num $gpu -datatype $datatype --random_seed $seed &> ${datatype}${seed}.out &
    done
done