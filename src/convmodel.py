from __future__ import division
from __future__ import print_function

import logging
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from util import get_minibatches, Progbar

tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("print_every", 100, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("train_dir", "../scratch", "Training directory to save the model parameters (default: ../scratch).")
FLAGS = tf.app.flags.FLAGS

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
        self.lr = 1e-3
        # lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
        #                                 staircase=True)
        self.session = tf.Session()
        self.options = options

    def close(self):
        self.session.close()

    def setup_placeholders(self, X_train):
        tp = tf.float32
        _,H,W,C = X_train.shape
        self.X_placeholder = tf.placeholder(tp, [None, H, W, C])
        self.y_placeholder = tf.placeholder(tp, [None,1])
        self.is_training = tf.placeholder(tf.bool)

    def setup_network(self, X, y, is_training):
        print("\n\n===== Setup Network ======\n\n")

        conv1_out = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
        bn1_out = tf.layers.batch_normalization(conv1_out, training=is_training)
        pool1_out = tf.layers.max_pooling2d(inputs=bn1_out, pool_size=[4, 4], strides=4)

        conv2_out = tf.layers.conv2d(inputs=pool1_out, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
        bn2_out = tf.layers.batch_normalization(conv2_out, training=is_training)
        pool2_out = tf.layers.max_pooling2d(inputs=bn2_out, pool_size=[4, 4], strides=4)

        flat_dim = np.product(pool2_out.shape[1:]).value
        pool2_out_flat = tf.reshape(pool2_out,[-1,flat_dim])
        affine1_out = tf.layers.dense(inputs=pool2_out_flat, units=1024, activation=tf.nn.relu)
        bn3_out = tf.layers.batch_normalization(affine1_out, training=is_training)
        dropout1_out = tf.layers.dropout(inputs=bn3_out, rate=0.4, training=is_training)

        affine2_out = tf.layers.dense(inputs=dropout1_out, units=512, activation=tf.nn.relu)
        bn4_out = tf.layers.batch_normalization(affine2_out, training=is_training)
        dropout2_out = tf.layers.dropout(inputs=bn4_out, rate=0.4, training=is_training)

        out_dim = np.product(y.shape[1:]).value
        affine3_out = tf.layers.dense(inputs=dropout2_out, units=out_dim, activation=tf.nn.relu)
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

        # gpu = self.options['gpu']
        # if gpu:
            # with tf.device('\gpu:0'):
                # self.pred = self.setup_network(self.X_placeholder, self.y_placeholder, self.is_training)
        # else:
        self.pred = self.setup_network(self.X_placeholder, self.y_placeholder, self.is_training)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        self.loss = tf.nn.l2_loss(self.pred - self.y_placeholder)

    def create_feed_dict(self, X, y, is_training=False):
        """
        Create a feed_dict
        :params: all are tensors of size [batch_size, ]
        :return: the feed_dict
        """

        feed_dict = {}
        feed_dict[self.X_placeholder] = X
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

        output_feed = [self.train_step, self.loss]

        _, loss = self.session.run(output_feed, feed_dict=input_feed)

        return loss

    def test(self, qns, mask_qns, cons, mask_cons, labels):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :param
        :return loss: validation loss for this batch
        """
        pass

        # TODO

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

    def run_epoch(self, X_train, y_train, X_val, y_val):
        """
        Run 1 epoch. Train on training examples, evaluate on validation set.
        """
        train_losses = []
        train_examples = [X_train, y_train]
        prog = Progbar(target=1 + int(len(X_train) / FLAGS.batch_size))
        for i, batch in enumerate(get_minibatches(train_examples, FLAGS.batch_size)):
            loss = self.optimize(*batch)
            train_losses.append(loss)
            if (self.global_step % FLAGS.print_every) == 0:
                logging.info("Iteration {0}: with minibatch training l2_loss = {1:.3g} and mse of {2:.2g}"\
                      .format(self.global_step, loss, loss/FLAGS.batch_size))
            prog.update(i + 1, [("train loss", loss)])
        total_train_mse = np.sum(train_losses)/X_train.shape[0]

        val_losses = []
        valid_examples = [X_val, y_val]
        prog = Progbar(target=1 + int(len(X_val) / FLAGS.batch_size))
        for i, batch in enumerate(get_minibatches(valid_examples, FLAGS.batch_size)):
            loss = self.validate(*batch)
            val_losses.append(loss)
            prog.update(i + 1, [("validation loss", loss)])
        logging.info("")
        total_val_mse = np.sum(val_losses)/X_val.shape[0]
        return total_train_mse, train_losses, total_val_mse, val_losses

    def train(self, X_train, X_val, vly_train, vly_val, agy_train, agy_val):
        """
        Implement main training loop

        :param
        :return:
        """

        plot_losses = self.options['plot_losses']

        self.setup_placeholders(X_train)
        self.setup_system()
        self.setup_loss()
        self.saver = tf.train.Saver(max_to_keep=50)

        # print number of parameters
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval(session=self.session)), params))
        logging.info("Number of params: %d" % num_params)

        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            grad_and_vars = optimizer.compute_gradients(self.loss)
            self.train_step = optimizer.apply_gradients(grad_and_vars, global_step=self.global_step)

        self.session.run(tf.global_variables_initializer())
        y_train = np.reshape(vly_train, (-1, 1))
        y_val = np.reshape(vly_val, (-1, 1))
        # training
        for epoch in range(FLAGS.epochs):
            logging.info("Epoch %d out of %d", epoch, FLAGS.epochs)
            total_train_mse, train_losses, total_val_mse, val_losses = \
                self.run_epoch(X_train, y_train, X_val, y_val)
            logging.info("Epoch {2}, Overall train mse = {0:.3g}, Overall val mse = {1:.3g}\n"\
                  .format(total_train_mse, total_val_mse, epoch))
            if plot_losses:
                plt.plot(train_losses)
                plt.plot(val_losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        # save model weights
        model_path = FLAGS.train_dir + "/convmodel_{:%Y%m%d_%H%M%S}_speedmode_{}/".format(datetime.now(), self.options['speedmode'])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logging.info("Saving model parameters...")
        self.saver.save(self.session, model_path + "model.weights")
        return (total_val_mse, 0, 0, 0)
