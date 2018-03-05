# MNIST Tuning

Tuning of hyperparameters. Use the shell scripts and just use file redirecting,
as in `python stuff.py > file.txt`.

## RMSProp Results

Some RMSProp observations after the coarse stage, with 55k training, best
validations:

1. lrate 0.001, wd 0.000001, v=1.275, t=1.402
2. lrate 0.001, wd 0.00001, v=1.285, t=1.462
3. lrate 0.005, wd 0.0, v=1.295, t=1.425
4. lrate 0.001, wd 0.0001, v=1.360, t=1.490
5. lrate 0.005, wd 0.000001, v=1.390, t=1.535

These are average values over 4 standard deviations. BTW.

Fine-tuned, ordered by validation ranks and one standard deviation:

1. lrate 0.001, wd 0.000001, v=1.259, t=1.404 (+/- 0.06) 
2. lrate 0.002, wd 0.000001, v=1.262, t=1.328 (+/- 0.05)
3. lrate 0.003, wd 0.000001, v=1.318, t=1.390 (+/- 0.08)
4. lrate 0.002, wd 0.00001,  v=1.320, t=1.453 (+/- 0.06)

So, seems like the coarse search was good enough. :-)

Now let's see what happens if we cheat and include the validation in training.
As a sanity check, the validation set performance is VERY good. But let's look
at the best test set:

1. lrate 0.001, wd 0.000001, t=1.367 (+/- 0.06) 
2. lrate 0.005, wd 0.0,      t=1.430 (+/- 0.06)

Wow, that's definitely better for (1) here, 1.367% error. These were run with 4
trials.


## Predicting the Target Image

(Not classification, but literally the digit.)

The command:

```
python targ_digit_test.py --num_train 2600 --num_epochs 1000 --lrate 0.001 --seed 0
```

results in:

```
Here are the variables in our network:
<tf.Variable 'cnn/conv2d/kernel:0' shape=(5, 5, 1, 32) dtype=float32_ref>
<tf.Variable 'cnn/conv2d/bias:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'cnn/conv2d_1/kernel:0' shape=(5, 5, 32, 64) dtype=float32_ref>
<tf.Variable 'cnn/conv2d_1/bias:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'cnn/dense/kernel:0' shape=(3136, 200) dtype=float32_ref>
<tf.Variable 'cnn/dense/bias:0' shape=(200,) dtype=float32_ref>
<tf.Variable 'cnn/dense_1/kernel:0' shape=(200, 200) dtype=float32_ref>
<tf.Variable 'cnn/dense_1/bias:0' shape=(200,) dtype=float32_ref>
<tf.Variable 'cnn/dense_2/kernel:0' shape=(200, 1) dtype=float32_ref>
<tf.Variable 'cnn/dense_2/bias:0' shape=(1,) dtype=float32_ref>
X_train.shape: (2600, 784)
y_train.shape: (2600, 1)
(End of debug prints)

------------------------
epoch | l2_loss | reg_loss | valid_acc | valid_diff |  test_acc | test_diff
    1    0.0029    7.0957      0.061      6.165      0.063      6.161
    2    0.0029    4.1634      0.152      3.864      0.165      3.775
    3    0.0029    2.9268      0.276      2.519      0.267      2.446
    4    0.0029    2.2577      0.351      1.927      0.359      1.924
    5    0.0029    1.9862      0.410      1.591      0.413      1.584
    6    0.0029    1.7185      0.424      1.458      0.431      1.454

...

  980    0.0070    0.0112      0.868      0.562      0.863      0.602
  981    0.0070    0.0091      0.870      0.539      0.860      0.596
  982    0.0070    0.0214      0.849      0.575      0.848      0.649
  983    0.0069    0.0076      0.849      0.608      0.845      0.640
  984    0.0069    0.0197      0.874      0.573      0.871      0.600
  985    0.0069    0.0184      0.865      0.541      0.867      0.580
  986    0.0069    0.0186      0.859      0.593      0.860      0.603
  987    0.0069    0.0083      0.862      0.580      0.858      0.609
  988    0.0069    0.0089      0.870      0.541      0.868      0.598
  989    0.0068    0.0122      0.859      0.561      0.852      0.625
  990    0.0068    0.0075      0.857      0.587      0.851      0.638
  991    0.0068    0.0110      0.874      0.564      0.871      0.594
  992    0.0068    0.0201      0.866      0.567      0.867      0.580
  993    0.0068    0.0205      0.843      0.662      0.843      0.639
  994    0.0068    0.0107      0.866      0.573      0.865      0.586
  995    0.0068    0.0088      0.874      0.536      0.870      0.573
  996    0.0068    0.0191      0.853      0.568      0.848      0.636
  997    0.0068    0.0190      0.839      0.600      0.836      0.657
  998    0.0068    0.0075      0.850      0.581      0.851      0.625
  999    0.0067    0.0106      0.869      0.591      0.863      0.602
 1000    0.0067    0.0228      0.844      0.631      0.846      0.619
```

wow, that's impressive! 85% prediction accuracy (if we count being accurate as
within 0.5 of the target digit) with just 2600 training digits. Seed 1 gives a
similar reward:

```
  988    0.0055    0.0017      0.881      0.551      0.877      0.624
  989    0.0054    0.0028      0.876      0.548      0.874      0.627
  990    0.0054    0.0040      0.873      0.550      0.871      0.634
  991    0.0054    0.0041      0.875      0.553      0.870      0.638
  992    0.0054    0.0033      0.875      0.555      0.871      0.637
  993    0.0053    0.0021      0.876      0.552      0.873      0.635
  994    0.0053    0.0014      0.879      0.547      0.876      0.626
  995    0.0053    0.0017      0.885      0.540      0.878      0.621
  996    0.0053    0.0037      0.884      0.541      0.879      0.620
  997    0.0052    0.0049      0.885      0.546      0.878      0.620
  998    0.0052    0.0046      0.886      0.555      0.877      0.626
  999    0.0052    0.0037      0.882      0.559      0.875      0.631
 1000    0.0052    0.0019      0.879      0.558      0.873      0.628
```
