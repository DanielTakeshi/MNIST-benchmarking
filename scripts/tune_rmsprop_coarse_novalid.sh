#!/bin/bash
size=400
for e in 0.01 0.005 0.001 0.0005 0.0001 ; do
    for w in 0.0 0.000001 0.00001 0.0001; do
        for (( i=1 ; i <= 4 ; i++ )); do
            python mnist_fc.py --optimizer rmsprop --fc_size $size --lrate $e --l2_reg $w --seed $i --no_validation > logs/rmsprop-tune/fc-$size-no-valid-lrate-$e-wd-$w-seed-$i
        done
    done
done
