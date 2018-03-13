# MNIST Tuning

Tuning of hyperparameters. Use the shell scripts and just use file redirecting,
as in `python stuff.py > file.txt`.

Contents:

- [RMSProp Results](#rmsprop-results)
- [Regressing on Target Image](#predicting-the-target-image)


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

### Full-Batch

I'm not doing classification, but literally predicting the digit, so the output
is a scalar. Here's what happens when you use the full data with "reasonable"
hyperparameters of batch size 100 and Adam learning rate 0.001:

![.](figures/mar06_numtrain_55500.png?raw=true)

So yes, it's a pretty easy task as predictive accuracy (being within 0.5 of the
correct target digit) is very high. Note that for the img1 and img2 L2 norms, I
forgot to take an average over those so the values are much larger than they
seem, and this also explains the discrepancy as there are 10k training images
and 5k validation images.

### 2.6k samples

For smaller minibatch sizes, the results will be slightly worse. Be careful that
you're using all the minibatches, by the way, so the batch size divides the
training data size. It might also be better to randomize the data after each
epoch with such a small dataset.

I did a minibatch size of 200 here, and with just 2600 data points. Here are the
results after 1000 epochs:

![.](figures/mar07_1000_epochs.png?raw=true)

Wow, the training accuracy did not even asympotote ... and to be clear we're not
memorizing anything, this is held-out validation and testing ... and clearly
learning rate of 0.001 is better in this case. I thought there would be some
more instability with just 2600 data points.

Here, I ran:

- 3000 epochs of Adam, 3 trials each.
- With shuffling the training dataset entirely at the start of each epoch, so
  this is almost like a true minibatch fashion (normally we have a fixed
  minibatch and iterate through them, I shuffle and then iterate through the
  resulting minibatches, but there's a third metric which involves iterating
  through an experience replay of samples which is probably best).
- I scaled the output into `[-0.5, 9.5]` when needed (`scale=True`) which means
  taking a hyperbolic tangent, then multiplying by 5, then adding by 4.5. It
  works.

![.](figures/mar12_3000_epochs.png?raw=true)

- Seems like we should realistically get 93 percent performance even without
  scaling, and scaling into the range gives even better performance. And recall,
  this is all with just 2600 training data points ...
- Oh, and we get almost as good performance based on 2000 data points (with
  scaling).

The plots are confusing in some ways, though. I can't explain the occasional
drop or jagged shape, and recall that this is with three trials.
