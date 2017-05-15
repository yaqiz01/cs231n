import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def run_model(session, is_training, X, y, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    print("\n\n===== START: run_model ======\n\n")

    # clear old variables
    tf.reset_default_graph()
    
    # have tensorflow compute l2 loss 
    l2_loss_val = tf.nn.l2_loss(predict - y)
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [l2_loss_val, loss_val]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        mse = 0.0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and mean squared error 
            # and (if given) perform a training step
            l2_loss, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(l2_loss)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and mse of {2:.2g}"\
                      .format(iter_cnt,loss,))
            iter_cnt += 1
        total_mse = np.sum(losses)/Xd.shape[0]
        print("Epoch {1}, Overall mse = {0:.3g}"\
              .format(total_mse,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_mse

def cnn(X, y, is_training):
    print("\n\n===== START: cnn ======\n\n")

    conv1_out = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
    bn1_out = tf.layers.batch_normalization(conv1_out, training=is_training)
    # 15 * 15 * 32
    pool1_out = tf.layers.max_pooling2d(inputs=bn1_out, pool_size=[2, 2], strides=2)
    
    conv2_out = tf.layers.conv2d(inputs=pool1_out, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
    bn2_out = tf.layers.batch_normalization(conv2_out, training=is_training)
    # 5 * 5 * 32
    pool2_out = tf.layers.max_pooling2d(inputs=bn2_out, pool_size=[2, 2], strides=2)
    
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
    
