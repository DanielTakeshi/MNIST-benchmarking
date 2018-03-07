for ((b=0; b<4; b++)); do
    python targ_digit_test.py --num_train 2600 --num_epochs 1000 --seed $b \
        --batch_size 200 --lrate 0.001 --cnn_arch 1
    python targ_digit_test.py --num_train 2600 --num_epochs 1000 --seed $b \
        --batch_size 200 --lrate 0.001 --cnn_arch 2
    python targ_digit_test.py --num_train 2600 --num_epochs 1000 --seed $b \
        --batch_size 200 --lrate 0.0001 --cnn_arch 2
done
