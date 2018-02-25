#!/bin/bash
size=400
for e in 0.001 0.002 0.003 0.004 0.005 ; do
    for w in 0.000001 0.00001 ; do
        for (( i=1 ; i<=20 ; i++ )) ; do
            python mnist_fc.py --optimizer rmsprop --fc_size $size --lrate $e --l2_reg $w --seed $i > logs/rmsprop-fine-tune/fc-$size-lrate-$e-wd-$w-seed-$i
        done
    done
done
