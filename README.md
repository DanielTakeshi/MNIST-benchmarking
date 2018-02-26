# MNIST Tuning

Tuning of hyperparameters. Use the shell scripts and just use file redirecting,
as in `python stuff.py > file.txt`.

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
