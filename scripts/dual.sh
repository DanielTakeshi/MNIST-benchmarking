for ((b=0; b<4; b++)); do
    python targ_digit_test.py --num_train 55000 --num_epochs 50 --seed $b \
        --batch_size 100 --lrate 0.001 --cnn_arch 1
    python targ_digit_test.py --num_train 55000 --num_epochs 50 --seed $b \
        --batch_size 100 --lrate 0.001 --cnn_arch 2
done
