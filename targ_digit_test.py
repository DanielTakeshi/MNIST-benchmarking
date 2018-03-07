""" Use this for not classification, but literally predicting the target digit. """
import argparse, logz, os, random, sys, time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import utils as U
np.set_printoptions(suppress=True, precision=4)


class Regressor:
    """ Only supports MNIST for now! """

    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        self.mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
        assert self.mnist.validation.labels.shape[0] == args.num_valid
        assert self.mnist.test.labels.shape[0] == args.num_test

        # All three datasets.
        self.X_train_all = self.mnist.train.images
        self.y_train_all = np.expand_dims(np.argmax(self.mnist.train.labels, axis=1), 1)
        self.X_valid = self.mnist.validation.images
        self.y_valid = np.expand_dims(np.argmax(self.mnist.validation.labels, axis=1), 1)
        self.X_test = self.mnist.test.images
        self.y_test = np.expand_dims(np.argmax(self.mnist.test.labels, axis=1), 1)

        # Test _subsets_ of the training data.
        inds = np.random.permutation(len(self.y_train_all))[:args.num_train]
        self.X_train = self.X_train_all[inds]
        self.y_train = self.y_train_all[inds]

        # Don't forget to add a second image to each training case.
        inds = np.random.permutation(len(self.y_train))[:args.num_train]
        self.X_train = np.concatenate((self.X_train, self.X_train_all[inds]), axis=1)

        # Placeholders, and build network.
        self.x      = tf.placeholder(tf.float32, [None, 784*2])
        self.y      = tf.placeholder(tf.float32, [None, 1])
        self.bnorm  = tf.placeholder(tf.bool, name='phase')
        self.layers = U.build_cnn(args=args, bn_train=self.bnorm, s=self.x)
        self.y_pred = self.layers['final']

        # Losses, etc.
        self.reg_loss = tf.reduce_mean( tf.square(self.y_pred-self.y) )
        self.variables = tf.trainable_variables()
        self.l2_loss = args.l2_reg * \
                tf.add_n([ tf.nn.l2_loss(v) for v in self.variables if 'bias' not in v.name ])
        self.loss = self.reg_loss + self.l2_loss

        # Optimization, etc.
        self.optimizer = U.get_optimizer(args)
        self.train_step = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.debug()


    def debug(self):
        print("\nHere are the variables in our network:")
        params = 0
        for item in self.variables:
            print(item)
            params += np.prod(item.get_shape().as_list())
        print("Total net parameters: {}\n".format(params))
        print("X_train.shape: {}".format(self.X_train.shape))
        print("y_train.shape: {}".format(self.y_train.shape))
        print("X_valid.shape: {}".format(self.X_valid.shape))
        print("y_valid.shape: {}".format(self.y_valid.shape))
        print("X_test.shape:  {}".format(self.X_test.shape))
        print("y_test.shape:  {}".format(self.y_test.shape))
        print("(End of debug prints)\n")

    
    def get_acc_diff(self, y_pred, y_targ):
        """ 
        Accuracy is defined as being within 0.5 of the correct target digit, so
        effectively we're rounding. Actually this also means getting, e.g., 9.6
        is wrong. We really want the values to be close. Note that our
        prediction network isn't guaranteed to be in the range (i.e. we don't
        always force a tanh/sigmoid on the output, for instance).
        """
        assert y_pred.shape == y_targ.shape and len(y_pred.shape) == 2
        acc = 0.0
        for (yp,yt) in zip(y_pred, y_targ):
            if np.abs(yp-yt) < 0.5:
                acc += 1.0
        diff = np.mean((y_pred-y_targ)**2)
        acc /= len(y_pred)
        return acc, diff


    def train(self):
        start_time = time.time()
        args = self.args
        feed_valid = {self.x: self.X_valid, self.y: self.y_valid, self.bnorm: 0}
        feed_test  = {self.x: self.X_test,  self.y: self.y_test,  self.bnorm: 0}
        train_stuff = [self.train_step, self.l2_loss, self.reg_loss]
        bs = args.batch_size
        num_mbs = int(args.num_train / bs)
        print("Training {} epochs, {} m-batches each".format(args.num_epochs, num_mbs))

        for epoch in range(1,args.num_epochs+1):
            for k in range(num_mbs):
                s = k * bs
                batch = (self.X_train[s:s+bs], self.y_train[s:s+bs])
                feed = {self.x: batch[0], self.y: batch[1], self.bnorm: 1}
                _, l2_loss, reg_loss = self.sess.run(train_stuff, feed)

            # Unlike training, for valid/test we'll randomize that second image.
            inds_v = np.random.permutation(args.num_valid)
            inds_t = np.random.permutation(args.num_test)
            feed_valid[self.x] = np.concatenate((self.X_valid, self.X_valid[inds_v]), axis=1)
            feed_test[self.x]  = np.concatenate((self.X_test,  self.X_test[inds_t]), axis=1)

            # We get the _layers_ for more fine-grained logging.
            layers_v = self.sess.run(self.layers, feed_valid)
            layers_t = self.sess.run(self.layers, feed_test)
            valid_acc, valid_diff = self.get_acc_diff(layers_v['final'], self.y_valid)
            test_acc,  test_diff  = self.get_acc_diff(layers_t['final'], self.y_test)

            print("\n  ************ After Epoch %i ************" % (epoch))
            elapsed_time_hours = (time.time() - start_time) / (60.0 ** 2)
            logz.log_tabular("ValidAvgAcc",  valid_acc)
            logz.log_tabular("TestAvgAcc",   test_acc)
            logz.log_tabular("ValidAvgDiff", valid_diff)
            logz.log_tabular("TestAvgDiff",  test_diff)
            logz.log_tabular("RegressLoss",  reg_loss)
            logz.log_tabular("L2RegLoss",    l2_loss)
            if args.cnn_arch == 2:
                logz.log_tabular("Img1AvgValL2",  np.linalg.norm(np.mean(layers_v['x1-branch'],axis=0)))
                logz.log_tabular("Img2AvgValL2",  np.linalg.norm(np.mean(layers_v['x2-branch'],axis=0)))
                logz.log_tabular("Img1AvgTestL2", np.linalg.norm(np.mean(layers_t['x1-branch'],axis=0)))
                logz.log_tabular("Img2AvgTestL2", np.linalg.norm(np.mean(layers_t['x2-branch'],axis=0)))
            logz.log_tabular("TrainEpochs",  epoch)
            logz.log_tabular("TimeHours",    elapsed_time_hours)
            logz.dump_tabular()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Bells and whistles, data management.
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--num_train', type=int, default=55000)
    parser.add_argument('--num_valid', type=int, default=5000)
    # Training and evaluation, stuff that should stay mostly constant:
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--momsgd_momentum', type=float, default=0.99)
    parser.add_argument('--rmsprop_decay', type=float, default=0.9)
    parser.add_argument('--rmsprop_momentum', type=float, default=0.0)
    # Training and evaluation, stuff to mostly tune:
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.00001)
    parser.add_argument('--optimizer', type=str, default='adam')
    # Network and data.
    parser.add_argument('--cnn_arch', type=int, default=1)
    parser.add_argument('--batch_norm', action='store_true')
    args = parser.parse_args()
    print("Our arguments:\n{}".format(args))

    logdir = 'logs/train-{}-epochs-{}-bsize-{}-arch-{}-bnorm-{}-seed-{}'.format(
        args.num_train, args.num_epochs, args.batch_size, args.cnn_arch,
        args.batch_norm, args.seed)
    print("logdir: {}\n".format(logdir))
    assert not os.path.exists(logdir), "error: {} exists!".format(logdir)
    logz.configure_output_dir(logdir)

    sess = U.get_tf_session(gpumem=0.8)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    regressor = Regressor(args, sess)
    regressor.train()
