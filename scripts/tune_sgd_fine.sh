#!/bin/bash
size=400
for e in 0.045 0.055 0.065 0.075; do
    for w in 0.0 0.00001; do
        for (( i=1 ; i<=20 ; i++ )); do
            python mnist_fc.py --optimizer sgd --fc_size $size --lrate $e --l2_reg $w --seed $i > logs/sgd-tune/fc-$size-lrate-$e-wd-$w-seed-$i
        done
    done
done
