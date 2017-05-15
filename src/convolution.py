import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def convolutionModel(X_train, X_test, vly_train, vly_test, agy_train, agy_test, **options):
    print("\n\n===== START: convolutionModel ======\n\n")
    learning_rate = 1e-3
    tp = tf.float32

    # setup input (e.g. the data that changes every batch)
    # The first dim is None, and gets sets automatically based on batch size fed in
    _,H,W,C = X_train.shape
    X = tf.placeholder(tp, [None, H, W, C])
    y = tf.placeholder(tp, [None,1])
    is_training = tf.placeholder(tf.bool)
    y_out = cnn(X, y ,is_training)

    #total_loss = tf.losses.mean_squared_error(y,y_out)
    mean_loss = tf.losses.mean_squared_error(y,y_out)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # batch normalization in tensorflow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)
    
    sess = tf.Session()
    
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval(session=sess)), params))
    print("Number of params: %d" % num_params)
    return

    sess.run(tf.global_variables_initializer())
    print('Training')
    y_train = np.reshape(vly_train, (-1, 1))
    run_model(sess,is_training,X,y,y_out,mean_loss,X_train,y_train,
            epochs=10,
            batch_size=8,
            print_every=1,
            training=train_step,
            plot_losses=False
            )
    print('Validation')
    y_test = np.reshape(vly_test, (-1, 1))
    vlmse = run_model(sess,is_training,X,y,y_out,mean_loss,X_test,y_test,
            epochs=1,
           batch_size=8
            )

    return (vlmse, 0.0, 0.0, 0.0)

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

    N = Xd.shape[0]
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        mse = 0.0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(N/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size) % N
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
                print("Iteration {0}: with minibatch training l2_loss = {1:.3g} and mse of {2:.2g}"\
                      .format(iter_cnt,l2_loss,l2_loss/batch_size))
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
    pool1_out = tf.layers.max_pooling2d(inputs=bn1_out, pool_size=[2, 2], strides=2)
    print('X.shape {}'.format(X.shape))
    print('bn1_out.shape {}'.format(bn1_out.shape))
    print('pool1_out.shape {}'.format(pool1_out.shape))
    
    conv2_out = tf.layers.conv2d(inputs=pool1_out, filters=32, kernel_size=[5, 5], activation=tf.nn.relu)
    bn2_out = tf.layers.batch_normalization(conv2_out, training=is_training)
    pool2_out = tf.layers.max_pooling2d(inputs=bn2_out, pool_size=[2, 2], strides=2)
    print('bn2_out.shape {}'.format(bn2_out.shape))
    print('pool2_out.shape {}'.format(pool2_out.shape))
    
    flat_dim = np.product(pool2_out.shape[1:]).value
    pool2_out_flat = tf.reshape(pool2_out,[-1,flat_dim])
    affine1_out = tf.layers.dense(inputs=pool2_out_flat, units=1024, activation=tf.nn.relu)
    bn3_out = tf.layers.batch_normalization(affine1_out, training=is_training)
    dropout1_out = tf.layers.dropout(inputs=bn3_out, rate=0.4, training=is_training)
    print('affine1_out.shape {}'.format(affine1_out.shape))
    print('bn3_out.shape {}'.format(bn3_out.shape))
    print('dropout1_out.shape {}'.format(dropout1_out.shape))
    
    affine2_out = tf.layers.dense(inputs=dropout1_out, units=512, activation=tf.nn.relu)
    bn4_out = tf.layers.batch_normalization(affine2_out, training=is_training)
    dropout2_out = tf.layers.dropout(inputs=bn4_out, rate=0.4, training=is_training)
    print('affine2_out.shape {}'.format(affine2_out.shape))
    print('bn4_out.shape {}'.format(bn4_out.shape))
    print('dropout2_out.shape {}'.format(dropout2_out.shape))
    
    out_dim = np.product(y.shape[1:]).value
    affine3_out = tf.layers.dense(inputs=dropout2_out, units=out_dim, activation=tf.nn.relu)
    print('affine3_out.shape {}'.format(affine3_out.shape))
    return affine3_out

def cnn_dummy(X, y, is_training):
    flat_dim = np.product(X.shape[1:]).value
    X_flat = tf.reshape(X,[-1,flat_dim])
    out_dim = np.product(y.shape[1:]).value
    out = tf.layers.dense(inputs=X_flat, units=out_dim, activation=tf.nn.relu)
    return out
