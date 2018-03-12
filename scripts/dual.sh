for ((b=2; b<3; b++)); do
    python targ_digit_test.py --num_train 2600 --num_epochs 3000 --seed $b --batch_size 200 --lrate 0.001  --cnn_arch 2
    python targ_digit_test.py --num_train 2600 --num_epochs 3000 --seed $b --batch_size 200 --lrate 0.0001 --cnn_arch 2
done
