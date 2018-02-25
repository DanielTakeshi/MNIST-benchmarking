#!/bin/bash
size=400
for e in 0.0001 0.0002 0.0003 0.0004 0.0005 ; do
    for w in 0.000001 0.00001 ; do
        for (( i=1 ; i<=20 ; i++ )) ; do
            python mnist_fc.py --optimizer rmsprop --fc_size $size --lrate $e --l2_reg $w --seed $i > logs/rmsprop-fine-tune/fc-$size-lrate-$e-wd-$w-seed-$i
        done
    done
done
