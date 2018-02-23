import logging
import utils
from dataprovider import Dataprovider
import tensorflow as tf
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


dataprovider = Dataprovider()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 65, 65, 3])
    y = tf.placeholder(tf.int32, [None, 5])


def inference(image):
    with tf.variable_scope('inference'):
        with tf.name_scope('1.unit'):
            w1 = utils.weight_variable([5, 5, 3, 32],name='w1')
            print(w1.name)
            b1 = utils.bias_variable([32], name='b1')
            tf.summary.histogram('b1', b1)
            x = utils.conv2d_activtion(image, w1, b1, batchnorm=True)
            x = utils.max_pool(x)

        with tf.name_scope('2.unit'):
            w2 = utils.weight_variable([5, 5, 32, 64], 'w2')
            tf.summary.histogram('w2', w2)
            b2 = utils.bias_variable([64], 'b2')
            tf.summary.histogram('b2', b2)
            x = utils.conv2d_activtion(x, w2, b2, batchnorm=True)
            x = utils.max_pool(x)

        with tf.name_scope('3.unit'):
            w3 = utils.weight_variable([5, 5, 64, 96], 'w3')
            b3 = utils.bias_variable([96], 'b3')
            x = utils.conv2d_activtion(x, w3, b3, batchnorm=True)
            x = utils.max_pool(x)

        with tf.name_scope('4.unit'):
            w4 = utils.weight_variable([5, 5, 96, 128], 'w4')
            b4 = utils.bias_variable([128], 'b4')
            x = utils.conv2d_activtion(x, w4, b4, batchnorm=True)
            x = utils.max_pool(x)

        with tf.name_scope('fc1'):
            layers = int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])
            w5 = utils.weight_variable([layers, 128], 'w5')
            b5 = utils.bias_variable([128], 'b5')
            x = tf.reshape(x, [-1, layers])
            x = tf.matmul(x, w5) + b5
            x = tf.nn.elu(x)
            # x = tf.nn.dropout(x, 0.5)

        with tf.name_scope('fc2'):
            w6 = utils.weight_variable([128, 128], 'w6')
            b6 = utils.bias_variable([128], 'b6')
            x = tf.matmul(x, w6) + b6
            x = tf.nn.elu(x)
            # x = tf.nn.dropout(x, 0.5)

        with tf.name_scope('out_put'):
            w7 = utils.weight_variable([128, 5], 'w7')
            b7 = utils.bias_variable([5], 'b7')
            x = tf.matmul(x, w7) + b7
            pred = tf.nn.softmax(x)

    return pred



def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer()
    grads = optimizer.compute_gradients(loss_val, var_list)
    return optimizer.apply_gradients(grads)


def main(iters=100000, batch_size=16):
    logits = inference(x)
    with tf.name_scope('loss'):
        loss =tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=y,
                                                                      name="entropy")))
    corr = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    with tf.name_scope('acc'):
        acc = tf.reduce_mean(tf.cast(corr, tf.float32))

    tf.summary.scalar('entropy', loss)
    tf.summary.scalar('acc', acc)
    summary_op = tf.summary.merge_all()

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    g = tf.get_default_graph()
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    test_writer = tf.summary.FileWriter('log/test')
    loss_tr, loss_test, acc_tr, acc_test = 0, 0, 0, 0
    for step in range(iters):
            print('权重w2之和：', np.sum(sess.run(g.get_tensor_by_name('inference/2.unit/w2:0'))))
            x_tr, y_tr = dataprovider.next_batch_tr(batch_size)
            x_test, y_test = dataprovider.next_batch_test(batch_size)
            list_tr = sess.run([loss, acc, summary_op, train_op],
                               feed_dict={x: x_tr, y: y_tr})
            list_test = sess.run([loss, acc, summary_op],
                                 feed_dict={x: x_test, y: y_test})
            train_writer.add_summary(list_tr[2], step)
            test_writer.add_summary(list_test[2], step)

            loss_tr += list_tr[0]
            acc_tr += list_tr[1]
            loss_test += list_test[0]
            acc_test += list_test[1]

            if step % 100 == 0:
                logging.info("step {:}, tr_loss: {:.4f}, tr_acc {:.4f}, test_loss: {:.4f}, test_acc {:.4f}"
                             .format(step, loss_tr / 10, acc_tr / 10, loss_test / 10, acc_test / 10))
                loss_tr, acc_tr, loss_test, acc_test = 0, 0, 0, 0



if __name__ == '__main__':
    main()