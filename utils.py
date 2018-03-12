import numpy as np
import tensorflow as tf
import sys

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


def get_optimizer(args):
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


def build_cnn(*, args, bn_train, s, osize=1, sname='mnist', renorm=True):
    """
    We assume the output size is 1, meaning that the result has shape (?,1).
    Returns a dictionary with final and intermediate layers, for intermediate
    inspection.
    """
    layers = {}
    bnorm = args.batch_norm
    assert len(s.shape) == 2
    x = tf.transpose(tf.reshape(s, [-1, 2, 28, 28]), [0, 2, 3, 1])
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    filt1 = 32
    filt2 = 64

    if args.cnn_arch == 1:
        # ----- ARCHITECTURE 1 -----
        with tf.variable_scope(sname):
            x = tf.keras.layers.Conv2D(filters=filt1, kernel_size=[5,5], padding='SAME')(x)
            x = tf.nn.relu(x)
            if bnorm:
                x = tf.layers.batch_normalization(x, training=bn_train, renorm=renorm)
            layers['conv-relu-1'] = x
            x = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x)
            x = tf.keras.layers.Conv2D(filters=filt2, kernel_size=[5,5], padding='SAME')(x)
            x = tf.nn.relu(x)
            if bnorm:
                x = tf.layers.batch_normalization(x, training=bn_train, renorm=renorm)
            layers['conv-relu-2'] = x
            x = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.nn.relu( tf.keras.layers.Dense(400)(x) )
            x = tf.nn.relu( tf.keras.layers.Dense(256)(x) )
            x = tf.keras.layers.Dense(osize)(x)
            layers['final'] = x
            return layers

    elif args.cnn_arch == 2:
        # ----- ARCHITECTURE 2 -----
        layers['x1-branch-start'] = x1
        layers['x2-branch-start'] = x2
        with tf.variable_scope(sname):
            # First image.
            x1 = tf.keras.layers.Conv2D(filters=filt1, kernel_size=[5,5], padding='SAME')(x1)
            x1 = tf.nn.relu(x1)
            if bnorm:
                x1 = tf.layers.batch_normalization(x1, training=bn_train, renorm=renorm)
            x1 = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x1)
            x1 = tf.keras.layers.Conv2D(filters=filt2, kernel_size=[5,5], padding='SAME')(x1)
            x1 = tf.nn.relu(x1)
            if bnorm:
                x1 = tf.layers.batch_normalization(x1, training=bn_train, renorm=renorm)
            x1 = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x1)
            x1 = tf.keras.layers.Flatten()(x1)
            x1 = tf.nn.relu( tf.keras.layers.Dense(400)(x1) )
            # Second image.
            x2 = tf.keras.layers.Conv2D(filters=filt1, kernel_size=[5,5], padding='SAME')(x2)
            x2 = tf.nn.relu(x2)
            if bnorm:
                x2 = tf.layers.batch_normalization(x2, training=bn_train, renorm=renorm)
            x2 = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x2)
            x2 = tf.keras.layers.Conv2D(filters=filt2, kernel_size=[5,5], padding='SAME')(x2)
            x2 = tf.nn.relu(x2)
            if bnorm:
                x2 = tf.layers.batch_normalization(x2, training=bn_train, renorm=renorm)
            x2 = tf.keras.layers.MaxPool2D(pool_size=[2,2])(x2)
            x2 = tf.keras.layers.Flatten()(x2)
            x2 = tf.nn.relu( tf.keras.layers.Dense(400)(x2) )
            # Concatenation.
            layers['x1-branch'] = x1
            layers['x2-branch'] = x2
            x = tf.concat([x1, x2], axis=1)
            layers['after-concat'] = x
            # Fully-connected.
            x = tf.nn.relu( tf.keras.layers.Dense(256)(x) )
            x = tf.keras.layers.Dense(osize)(x)
            if args.scale_output:
                # This is specific to this case, bad but might as well test...
                x = (tf.nn.tanh(x) * 5.0) + 4.5
            layers['final'] = x
            return layers

    else:
        raise ValueError(args.cnn_arch)
