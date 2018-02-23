import tensorflow as tf

def conv2d_activtion(x, w, b, batchnorm=True):
    net = conv2d(x, w, b)

    if batchnorm == True:
        net = norm(net)
    return tf.nn.elu(net)



def norm(x):
    return tf.layers.batch_normalization(x)


def weight_variable(shape, name, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def conv2d(x, W, b, padding='SAME'):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    return conv_2d + b


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')