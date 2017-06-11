from __future__ import division
from __future__ import print_function

from _init_paths import *
import logging
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from speeddet import *
from util import get_minibatches, Progbar
from play import *

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout rate.")
tf.app.flags.DEFINE_integer("epochs", 15, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("decay_step", 100, "Number of steps between decays.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Decay rate.")
tf.app.flags.DEFINE_integer("print_every", 100, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("weight_init", "xavier", "tf method for weight initialization")
tf.app.flags.DEFINE_string("step_optimize", False, "whether to optimize separately")
FLAGS = tf.app.flags.FLAGS
FLAGS._parse_flags()
flagDict = FLAGS.__dict__['__flags']
for flag in flagDict:
    print('Configuration: {}={}'.format(flag, flagDict[flag]))

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(alpha * x, x)

def baseline(X, is_training):
    conv1_out = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
    bn1_out = tf.layers.batch_normalization(conv1_out, training=is_training)
    pool1_out = tf.layers.max_pooling2d(inputs=bn1_out, pool_size=[4, 4], strides=4)

    conv2_out = tf.layers.conv2d(inputs=pool1_out, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
    bn2_out = tf.layers.batch_normalization(conv2_out, training=is_training)
    pool2_out = tf.layers.max_pooling2d(inputs=bn2_out, pool_size=[4, 4], strides=4)

    return pool2_out

def res_block(X, num_filters, is_training, downsample=False):
    stride = 1
    if downsample:
        stride = 2
        num_filters *= 2
    conv1 = tf.layers.conv2d(X, num_filters, 3, strides=stride, padding='same', activation=tf.nn.relu)
    bn1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv2 = tf.layers.conv2d(bn1, num_filters, 3, padding='same', activation=tf.nn.relu)
    bn2 = tf.layers.batch_normalization(conv2, training=is_training)
    if downsample:
        X = tf.layers.average_pooling2d(X, stride, stride, padding='same')
        X = tf.pad(X, [[0,0], [0,0], [0,0], [num_filters // 4, num_filters // 4]])
    return bn2 + X

def res_net(X, is_training):
    init_out = tf.layers.conv2d(X, 64, 7, strides=2, activation=tf.nn.relu)
    layer_in = tf.layers.batch_normalization(init_out, training=is_training)
    for i in range(2):
        out = res_block(layer_in, 64, is_training)
        layer_in = out
    num_filters = [64, 128, 256]
    block_in = out
    for num_filter in num_filters:
        layer_1 = res_block(block_in, num_filter, is_training, downsample=True)
        layer_2 = res_block(layer_1, 2 * num_filter, is_training)
        block_in = layer_2
    return layer_2

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def alex_net(X, is_training):
    init = tf.contrib.layers.xavier_initializer()
    if FLAGS.weight_init == "trunc_normal":
	    init = trunc_normal(0.0005)

    with tf.variable_scope("conv_1"):
        X = tf.layers.conv2d(X, 64, 11, strides=4, activation=tf.nn.relu, kernel_initializer=init)
        X = tf.layers.max_pooling2d(X, 3, 2)
    with tf.variable_scope("conv_2"):
        X = tf.layers.conv2d(X, 192, 5, activation=tf.nn.relu, kernel_initializer=init)
        X = tf.layers.max_pooling2d(X, 3, 2)
    with tf.variable_scope("conv_3"):
        X = tf.layers.conv2d(X, 384, 3, activation=tf.nn.relu, kernel_initializer=init)
    with tf.variable_scope("conv_4"):
        X = tf.layers.conv2d(X, 384, 3, activation=tf.nn.relu, kernel_initializer=init)
    with tf.variable_scope("conv_5"):
        X = tf.layers.conv2d(X, 256, 3, activation=tf.nn.relu, kernel_initializer=init)
        X = tf.layers.max_pooling2d(X, 3, 2)
    with tf.variable_scope("fc_6"):
        X = tf.layers.conv2d(X, 4096, 5, activation=tf.nn.relu, kernel_initializer=init)
        X = tf.layers.dropout(X, rate=FLAGS.dropout, training=is_training)
    with tf.variable_scope("fc_7"):
        X = tf.layers.conv2d(X, 4096, 1, activation=tf.nn.relu, kernel_initializer=init)
        X = tf.layers.dropout(X, rate=FLAGS.dropout, training=is_training)
    return X

def conv_helper(input, kernel, biases, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        print("input is: ", input)
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        print("kernel is: ", kernel)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def alexnet_v2_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc

def alexnet_v2(X, is_training, scope='alexnet_v2'):
  with tf.variable_scope(scope, 'alexnet_v2', [X]) as sc:
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
      net = slim.conv2d(X, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3')
      net = slim.conv2d(net, 384, [3, 3], scope='conv4')
      net = slim.conv2d(net, 256, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=trunc_normal(0.005),
                          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6')
        net = slim.dropout(net, FLAGS.dropout, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, FLAGS.dropout, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, 4096, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='fc8')

      return net

class ConvModel(object):
    def __init__(self, options):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up training/updating procedure ====
        # implement learning rate annealing
        if options['convmode'] == 4:
	    self.pretrained_weights = np.load('bvlc_alexnet.npy')[()]
        self.session = tf.Session()
        self.options = options
        self.setup_placeholders(**options)
        self.setup_system()
        self.setup_loss()
        self.saver = tf.train.Saver(max_to_keep=50)

    def close(self):
        self.session.close()

    def setup_placeholders(self, **options):
        flowmode = options['flowmode']
        rseg = options['rseg']
        cseg = options['cseg']
        tp = tf.float32
        H,W,C = options['inputshape']
        if flowmode in [2,3]:
            H = rseg; W = cseg;
        self.X_placeholder = tf.placeholder(tp, [None, H, W, C])
        self.y_placeholder = tf.placeholder(tp, [None,3])
        self.is_training = tf.placeholder(tf.bool)

    def alex_net_pretrained(self, X, is_training):
        init = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("conv_1"):
            X = tf.layers.conv2d(X, 96, 11, strides=4, activation=tf.nn.relu, kernel_initializer=init)
            X = tf.layers.max_pooling2d(X, 3, 2)
        with tf.variable_scope("conv_2"):
            conv2_W = tf.Variable(self.pretrained_weights['conv2'][0])
            conv2_b = tf.Variable(self.pretrained_weights['conv2'][1])
            X = conv_helper(X, conv2_W, conv2_b, group=2)
            X = tf.layers.max_pooling2d(X, 3, 2)
        with tf.variable_scope("conv_3"):
            conv3_W = tf.Variable(self.pretrained_weights['conv3'][0])
            conv3_b = tf.Variable(self.pretrained_weights['conv3'][1])
            X = conv_helper(X, conv3_W, conv3_b, group=1)
        with tf.variable_scope("conv_4"):
            conv4_W = tf.Variable(self.pretrained_weights['conv4'][0])
            conv4_b = tf.Variable(self.pretrained_weights['conv4'][1])
            X = conv_helper(X, conv4_W, conv4_b, group=2)
        with tf.variable_scope("conv_5"):
            conv5_W = tf.Variable(self.pretrained_weights['conv5'][0])
            conv5_b = tf.Variable(self.pretrained_weights['conv5'][1])
            X = conv_helper(X, conv5_W, conv5_b, group=2)
            X = tf.layers.max_pooling2d(X, 3, 2)
        with tf.variable_scope("fc_6"):
            X = tf.layers.conv2d(X, 4096, 5, activation=tf.nn.relu, kernel_initializer=init)
            X = tf.layers.dropout(X, rate=FLAGS.dropout, training=is_training)
        with tf.variable_scope("fc_7"):
            X = tf.layers.conv2d(X, 4096, 1, activation=tf.nn.relu, kernel_initializer=init)
            X = tf.layers.dropout(X, rate=FLAGS.dropout, training=is_training)
        return X

    def setup_network(self, X, y, is_training):
        print("\n\n===== Setup Network ======\n\n")

        if self.options['convmode'] == 0:
            baseline_out = baseline(X, is_training)
            flat_dim = np.product(baseline_out.shape[1:]).value
            baseline_out_flat = tf.reshape(baseline_out, [-1, flat_dim])

            affine1_out = tf.layers.dense(inputs=baseline_out_flat, units=1024, activation=tf.nn.relu)
            bn3_out = tf.layers.batch_normalization(affine1_out, training=is_training)
            dropout1_out = tf.layers.dropout(inputs=bn3_out, rate=0.4, training=is_training)

            affine2_out = tf.layers.dense(inputs=dropout1_out, units=512, activation=tf.nn.relu)
            bn4_out = tf.layers.batch_normalization(affine2_out, training=is_training)
            conv_out = tf.layers.dropout(inputs=bn4_out, rate=0.4, training=is_training)

        elif self.options['convmode'] == 1:
            res_out = res_net(X, is_training)
            flat_dim = np.product(res_out.shape[1:]).value
            conv_out = tf.reshape(res_out, [-1, flat_dim])

        elif self.options['convmode'] == 2:
            alex_out = alex_net(X, is_training)
            flat_dim = np.product(alex_out.shape[1:]).value
            conv_out = tf.reshape(alex_out, [-1, flat_dim])

        elif self.options['convmode'] == 3:
            with slim.arg_scope(alexnet_v2_arg_scope()):
                alex_out = alexnet_v2(X, is_training)
                flat_dim = np.product(alex_out.shape[1:]).value
                conv_out = tf.reshape(alex_out, [-1, flat_dim])

        elif self.options['convmode'] == 4:
            alex_out = self.alex_net_pretrained(X, is_training)
            flat_dim = np.product(alex_out.shape[1:]).value
            conv_out = tf.reshape(alex_out, [-1, flat_dim])

        out_dim = np.product(y.shape[1:]).value
        affine3_out = tf.layers.dense(inputs=conv_out, units=out_dim, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return affine3_out

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :param qn_embeddings: embedding of question words, of size (batch_size, embedding_size)
        :param con_embeddings: embedding of context words, of size (batch_size, embedding_size)
        :self.start_pred & self.end_pred: tensors of shape [batch_size, FLAGS.con_max_len]
                                          a probability distribution over context
        """

        cpu = self.options['cpu']
        if cpu:
            with tf.device('\cpu:0'):
                self.pred = self.setup_network(self.X_placeholder, self.y_placeholder, self.is_training)
        else:
            self.pred = self.setup_network(self.X_placeholder, self.y_placeholder, self.is_training)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        self.loss = tf.nn.l2_loss(self.pred - self.y_placeholder)
        if FLAGS.step_optimize :
            self.loss1 = tf.slice(self.pred - self.y_placeholder, [0,0], [-1, 1])
            self.loss2 = tf.slice(self.pred - self.y_placeholder, [0,1], [-1, 1])
            self.loss3 = tf.slice(self.pred - self.y_placeholder, [0,2], [-1, 1])

    def create_feed_dict(self, X, y, is_training=False):
        """
        Create a feed_dict
        :params: all are tensors of size [batch_size, ]
        :return: the feed_dict
        """

        feed_dict = {}
        feed_dict[self.X_placeholder] = X
        if y is not None:
            feed_dict[self.y_placeholder] = y
        feed_dict[self.is_training] = is_training

        return feed_dict

    def optimize(self, X_train, y_train):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :param
        :return loss: training loss
        """
        # construct feed_dict
        input_feed = self.create_feed_dict(X_train, y_train, True)

        if FLAGS.step_optimize :
            output_feed = [self.train_step1, self.train_step2, self.train_step3, self.loss, self.lr, self.global_step]
            _, _, _, loss, lr, gs = self.session.run(output_feed, feed_dict=input_feed)
            return loss, lr, gs
        else:
            output_feed = [self.train_step, self.loss, self.lr, self.global_step]
            _, loss, lr, gs = self.session.run(output_feed, feed_dict=input_feed)
            return loss, lr, gs

    def get_model_path(self):
        return SCRATCH_PATH + ("convmodel_speedmode_{}_convmode_{}/".format(self.options['speedmode'], self.options['convmode']))

    def restore(self):
        model_path = self.get_model_path()
        if not os.path.exists(model_path):
            print('Error! Do not have saved parameter for {}'.format(model_path))
            sys.exit(-1)
        logging.info("Loading model parameters from {}".format(model_path))
        self.saver.restore(self.session, model_path + "model.weights")

    def test(self, X_test):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :param
        :return loss: validation loss for this batch
        """

        # compute loss for every single example and add together
        input_feed = self.create_feed_dict(np.array([X_test]), None, is_training=False)

        output_feed = self.pred

        y = self.session.run(output_feed, feed_dict=input_feed)
        vf = y[0, 0]
        wu = y[0, 1]
        af = y[0, 2]

        return vf, wu, af

    def validate(self, X_val, y_val):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :param
        :return: validation cost of the batch
        """

        # compute loss for every single example and add together
        input_feed = self.create_feed_dict(X_val, y_val)

        output_feed = self.loss

        loss = self.session.run(output_feed, feed_dict=input_feed)

        return loss

    def run_epoch(self, frameTrain, frameVal):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        """
        options = self.options
        path = options['path']
        train_losses = []
        numTrain = frameTrain.shape[0]
        prog = Progbar(target=1 + int(numTrain / FLAGS.batch_size))
        for i, frameBatch in enumerate(get_minibatches(frameTrain, FLAGS.batch_size)):
            batch = loadData(frameBatch, **options)
            loss, lr, gs = self.optimize(*batch)
            train_losses.append(loss)
            if (self.global_step % FLAGS.print_every) == 0:
                logging.info("Iteration {0}: with minibatch training l2_loss = {1:.3g} and mse of {2:.2g}"\
                      .format(self.global_step, loss, loss/FLAGS.batch_size))
            prog.update(i + 1, [("train loss", loss)], [("learning rate", lr), ("global step", gs)])
        total_train_mse = np.sum(train_losses)/numTrain

        val_losses = []
        numVal = frameVal.shape[0]
        prog = Progbar(target=1 + int(numVal / FLAGS.batch_size))
        for i, frameBatch in enumerate(get_minibatches(frameVal, FLAGS.batch_size)):
            batch = loadData(frameBatch, **options)
            loss = self.validate(*batch)
            val_losses.append(loss)
            prog.update(i + 1, [("validation loss", loss)])
        total_val_mse = np.sum(val_losses)/numVal
        return total_train_mse, train_losses, total_val_mse, val_losses

    def train(self, frameTrain, frameVal):
        """
        Implement main training loop

        :param
        :return:
        """

        plot_losses = self.options['plot_losses']

        # print number of parameters
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval(session=self.session)), params))
        logging.info("Number of params: %d" % num_params)

        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=True)
            self.lr = lr
            if FLAGS.step_optimize :
                optimizer1 = tf.train.AdamOptimizer(self.lr)
                optimizer2 = tf.train.AdamOptimizer(self.lr)
                optimizer3 = tf.train.AdamOptimizer(self.lr)

                grad_and_vars1 = optimizer1.compute_gradients(self.loss1)
                self.train_step1 = optimizer1.apply_gradients(grad_and_vars1, global_step=self.global_step)
                grad_and_vars2 = optimizer2.compute_gradients(self.loss2)
                self.train_step2 = optimizer2.apply_gradients(grad_and_vars2, global_step=self.global_step)
                grad_and_vars3 = optimizer2.compute_gradients(self.loss3)
                self.train_step3 = optimizer2.apply_gradients(grad_and_vars3, global_step=self.global_step)
            else :
                optimizer = tf.train.AdamOptimizer(self.lr)
                grad_and_vars = optimizer.compute_gradients(self.loss)
                self.train_step = optimizer.apply_gradients(grad_and_vars, global_step=self.global_step)

        self.session.run(tf.global_variables_initializer())
        # y_train = np.reshape(vly_train, (-1, 1))
        # y_val = np.reshape(vly_val, (-1, 1))
        # training
        min_val_mse = sys.maxint
        for epoch in range(FLAGS.epochs):
            logging.info("Epoch %d out of %d", epoch+1, FLAGS.epochs)
            total_train_mse, train_losses, total_val_mse, val_losses = \
                self.run_epoch(frameTrain, frameVal)
            logging.info("Epoch {2}, Overall train mse = {0:.4g}, Overall val mse = {1:.4g}\n"\
                  .format(total_train_mse, total_val_mse, epoch+1))
            # save model weights
            if total_val_mse < min_val_mse:
                min_val_mse = total_val_mse
                model_path = self.get_model_path()
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                logging.info("Saving model parameters of epoch {}...".format(epoch+1))
                self.saver.save(self.session, model_path + "model.weights")
            if plot_losses:
                plt.plot(train_losses)
                plt.plot(val_losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        # return (total_val_mse, 0, 0, 0)
