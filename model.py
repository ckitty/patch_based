import logging
import tensorflow as tf
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def inference(x):
    # x = x/255.0
    with tf.name_scope('1.unit'):
        # w1 = weight_variable([5, 5, 3, 32],name='w1')
        a = np.ones([5,5,3,32],dtype=np.float32)
        w1 = tf.Variable(a, name='w1')
        print(w1.name)
        tf.summary.histogram('w1',w1)
        b1 = bias_variable([32],name='b1')
        tf.summary.histogram('b1',b1)
        x = conv2d_activtion(x, w1, b1, batchnorm=True)
        x = max_pool(x)

    with tf.name_scope('2.unit'):
        w2 = weight_variable([5, 5, 32, 64],'w2')
        tf.summary.histogram('w2',w2)
        b2 = bias_variable([64],'b2')
        tf.summary.histogram('b2',b2)
        x = conv2d_activtion(x, w2, b2, batchnorm=True)
        x = max_pool(x)

    with tf.name_scope('3.unit'):
        w3 = weight_variable([5, 5, 64, 96],'w3')
        b3 = bias_variable([96],'b3')
        x = conv2d_activtion(x, w3, b3, batchnorm=True)
        x = max_pool(x)

    with tf.name_scope('4.unit'):
        w4 = weight_variable([5, 5, 96, 128],'w4')
        b4 = bias_variable([128],'b4')
        x = conv2d_activtion(x, w4, b4, batchnorm=True)
        x = max_pool(x)

    with tf.name_scope('fc1'):
        layers = int(x.shape[1])*int(x.shape[2])*int(x.shape[3])
        w5 = weight_variable([layers, 128],'w5')
        b5 = bias_variable([128],'b5')
        x = tf.reshape(x, [-1, layers])
        x = tf.matmul(x, w5) + b5
        x = tf.nn.elu(x)
        # x = tf.nn.dropout(x, 0.5)

    with tf.name_scope('fc2'):
        w6 = weight_variable([128, 128],'w6')
        b6 = bias_variable([128],'b6')
        x = tf.matmul(x, w6) + b6
        x = tf.nn.elu(x)
        # x = tf.nn.dropout(x, 0.5)

    with tf.name_scope('out_put'):
        w7 = weight_variable([128, 5],'w7')
        b7 = bias_variable([5],'b7')
        x = tf.matmul(x, w7) + b7
        pred = tf.nn.softmax(x)

    return pred





class Model(object):
    def __init__(self, dataprovider, batch_size):
        self.dataprovider = dataprovider
        self.x = tf.placeholder(tf.float32, [None, 65, 65, 3])
        self.logits = inference(self.x)
        self.y = tf.placeholder(tf.int32, [None, 5])


        # self.loss = tf.losses.softmax_cross_entropy(self.y,logits)

        corr = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))  # 对比预测值的索引和真实label的索引是否一样，一样返回True，不一样返回False
        self.accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))

        self.batch_size = batch_size



    # def initialize(self, sess):
    #     tf.summary.scalar('loss',self.loss)
    #     tf.summary.scalar('accuracy',self.accuracy)
    #     self.summary_op = tf.summary.merge_all()
    #     self.train_writer = tf.summary.FileWriter('log/train', sess.graph)
    #     self.test_writer = tf.summary.FileWriter('log/test')
    #     self.optimizer = tf.train.AdamOptimizer().minimize(self.loss,var_list=tf.trainable_variables())
    #     sess.run(tf.global_variables_initializer())


    def train(self, iters=100000):
        logging.info("Start optimization")
        g = tf.get_default_graph()
        print(g.get_all_collection_keys())
        list = g.get_collection('trainable_variables')
        for i in list:
            print(i)
        loss = tf.losses.softmax_cross_entropy(self.y,self.logits)
        # lossss = g.get_collection('losses')
        # print(lossss)
        with tf.Session() as sess:
            # self.initialize(sess)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('log/train', sess.graph)
            self.test_writer = tf.summary.FileWriter('log/test')
            self.optimizer = tf.train.AdamOptimizer().minimize(loss)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            loss_tr,loss_test = 0, 0
            acc_tr, acc_test = 0, 0
            for step in range(iters):
                print('权重w1之和：', np.sum(sess.run(g.get_tensor_by_name('1.unit/w1:0'))))
                x_tr, y_tr = self.dataprovider.next_batch_tr(self.batch_size)
                x_test, y_test = self.dataprovider.next_batch_test(self.batch_size)
                list_tr = sess.run([loss, self.accuracy, self.summary_op, self.optimizer],
                                   feed_dict={self.x:x_tr, self.y:y_tr})
                list_test = sess.run([loss, self.accuracy, self.summary_op],
                                     feed_dict={self.x:x_test, self.y:y_test})
                self.train_writer.add_summary(list_tr[2], step)
                self.test_writer.add_summary(list_test[2], step)

                loss_tr += list_tr[0]
                acc_tr += list_tr[1]
                loss_test += list_test[0]
                acc_test += list_test[1]

                if step%10 == 0:
                    logging.info("step {:}, tr_loss: {:.4f}, tr_acc {:.4f}, test_loss: {:.4f}, test_acc {:.4f}"
                                 .format(step, loss_tr/10, acc_tr/10, loss_test/10, acc_test/10))
                    loss_tr, acc_tr, loss_test, acc_test = 0, 0, 0, 0





























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