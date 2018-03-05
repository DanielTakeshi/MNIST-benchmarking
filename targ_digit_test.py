""" Use this for not classification, but literally predicting the target digit. """
import argparse, random, sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True, precision=4)

def get_tf_session(gpumem):
    """ Returning a session. Set options here if desired. """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpumem)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


class Regressor:
    """ Only supports MNIST for now! """

    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        self.mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
        assert self.mnist.validation.labels.shape[0] == args.num_valid
        assert self.mnist.test.labels.shape[0] == args.num_test

        # The data
        self.X_train = self.mnist.train.images
        self.y_train = np.expand_dims(np.argmax(self.mnist.train.labels, axis=1), 1)
        self.X_valid = self.mnist.validation.images
        self.y_valid = np.expand_dims(np.argmax(self.mnist.validation.labels, axis=1), 1)
        self.X_test  = self.mnist.test.images
        self.y_test  = np.expand_dims(np.argmax(self.mnist.test.labels, axis=1), 1)

        # Test subsets of the training data.
        inds = np.random.permutation(len(self.y_train))[:args.num_train]
        self.X_train = self.X_train[inds]
        self.y_train = self.y_train[inds]

        # Placeholders, network output, losses.
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.y_pred = self.make_network(self.x)
        self.reg_loss = tf.reduce_mean( tf.square(self.y_pred-self.y) )
        self.variables = tf.trainable_variables()
        self.l2_loss = args.l2_reg * \
                tf.add_n([ tf.nn.l2_loss(v) for v in self.variables if 'bias' not in v.name ])
        self.loss = self.reg_loss + self.l2_loss

        # Optimization, etc.
        self.optimizer = self.get_optimizer()
        self.train_step = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.debug()


    def get_optimizer(self):
        args = self.args
        name = (args.optimizer).lower()
        if name == 'sgd':
            return tf.train.GradientDescentOptimizer(args.lrate)
        elif name == 'momsgd':
            return tf.train.MomentumOptimizer(args.lrate,
                    momentum=args.momsgd_momentum)
        elif name == 'rmsprop':
            return tf.train.RMSPropOptimizer(args.lrate,
                    decay=args.rmsprop_decay,
                    momentum=args.rmsprop_momentum)
        elif name == 'adam':
            return tf.train.AdamOptimizer(args.lrate)
        else:
            raise ValueError()


    def make_network(self, x):
        if self.args.net_type == 'ff':
            return self.make_ff(x)
        elif self.args.net_type == 'cnn':
            return self.make_cnn(x)
        else:
            raise ValueError()


    def debug(self):
        print("Here are the variables in our network:")
        for item in self.variables:
            print(item)
        print("X_train.shape: {}".format(self.X_train.shape))
        print("y_train.shape: {}".format(self.y_train.shape))
        print("(End of debug prints)\n")


    def make_ff(self, x):
        size = self.args.fc_size
        with tf.variable_scope('ff'):
            x = tf.nn.relu( tf.keras.layers.Dense(size)(x) )
            x = tf.nn.relu( tf.keras.layers.Dense(size)(x) )
            x = tf.keras.layers.Dense(1)(x)
            return x


    def make_cnn(self, x):
        x = tf.transpose(tf.reshape(x, [-1, 1, 28, 28]), [0, 2, 3, 1])
        with tf.variable_scope('cnn'):
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='SAME')(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x) # shape = (?, 14, 14, 32)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=[5,5], padding='SAME')(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x) # shape = (?, 7, 7, 64)
            x = tf.keras.layers.Flatten()(x) # shape = (?, 7*7*64) = (?, 3136)
            x = tf.nn.relu( tf.keras.layers.Dense(200)(x) )
            x = tf.nn.relu( tf.keras.layers.Dense(200)(x) )
            x = tf.keras.layers.Dense(1)(x)
            return x

    
    def get_acc_diff(self, y_pred, y_targ):
        assert y_pred.shape == y_targ.shape and len(y_pred.shape) == 2
        acc = 0.0
        for (yp,yt) in zip(y_pred, y_targ):
            if np.abs(yp-yt) < 0.5:
                acc += 1.0
        diff = np.mean((y_pred-y_targ)**2)
        acc /= len(y_pred)
        return acc, diff


    def train(self):
        args = self.args
        mnist = self.mnist
        feed_valid = {self.x: self.X_valid, self.y: self.y_valid}
        feed_test  = {self.x: self.X_test,  self.y: self.y_test}
        print('------------------------')
        print("epoch | l2_loss | reg_loss | valid_acc | valid_diff |  test_acc | test_diff")
        bs = args.batch_size
        train_stuff = [self.train_step, self.l2_loss, self.reg_loss]

        for epoch in range(1,args.num_epochs+1):
            num_mbs = int(args.num_train / bs)

            for k in range(num_mbs):
                s = k * bs
                batch = (self.X_train[s:s+bs], self.y_train[s:s+bs])
                feed = {self.x: batch[0], self.y: batch[1]}
                _, l2_loss, reg_loss = self.sess.run(train_stuff, feed)

            y_pred_valid = self.sess.run(self.y_pred, feed_valid)
            y_pred_test  = self.sess.run(self.y_pred, feed_test)
            valid_acc, valid_diff = self.get_acc_diff(y_pred_valid, self.y_valid)
            test_acc, test_diff   = self.get_acc_diff(y_pred_test, self.y_test)
            #print(y_pred_test[:20].T)
            #print(self.y_test[:20].T)
            print("{:5} {:9.4f} {:9.4f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}".format(
                    epoch, l2_loss, reg_loss, valid_acc, valid_diff, test_acc, test_diff)
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Bells and whistles
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data')
    parser.add_argument('--seed', type=int, default=1)
    # Training and evaluation, stuff that should stay mostly constant:
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--momsgd_momentum', type=float, default=0.99)
    parser.add_argument('--rmsprop_decay', type=float, default=0.9)
    parser.add_argument('--rmsprop_momentum', type=float, default=0.0)
    # Training and evaluation, stuff to mostly tune:
    parser.add_argument('--lrate', type=float, default=0.0001)
    parser.add_argument('--l2_reg', type=float, default=0.00001)
    parser.add_argument('--optimizer', type=str, default='adam')
    # Network and data. the 784-400-400-10 seems a common benchmark.
    parser.add_argument('--fc_size', type=int, default=400)
    parser.add_argument('--net_type', type=str, default='cnn')
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--num_train', type=int, default=55000)
    parser.add_argument('--num_valid', type=int, default=5000)
    args = parser.parse_args()
    print("Our arguments:\n{}".format(args))

    sess = get_tf_session(gpumem=0.8)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    regressor = Regressor(args, sess)
    regressor.train()
