import tensorflow as tf
import numpy as np


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
#
# he_init = tf.contrib.layers.variance_scaling_initializer()
#
# #Function for building the DNN
# def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None,
#         activation=tf.nn.elu, initializer=he_init):
#     with tf.variable_scope(name, "dnn"):
#         for layer in range(n_hidden_layers):
#             inputs = tf.layers.dense(inputs, n_neurons, activation=activation,
#                                      kernel_initializer=initializer,
#                                      name="hidden%d" % (layer + 1))
#         return inputs
#
#
# n_inputs = 28 * 28 # MNIST
# n_outputs = 5
#
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y = tf.placeholder(tf.int64, shape=(None), name="y")
#
# dnn_outputs = dnn(X)
#
# logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
# Y_proba = tf.nn.softmax(logits, name="Y_proba")
#
#
# #Adam Optimization code
# learning_rate = 0.01
#
# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
# loss = tf.reduce_mean(xentropy, name="loss")
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss, name="training_op")
#
# correct = tf.nn.in_top_k(logits, y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
#MNIST Data Prcessing
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
#
#
# X_train1 = mnist.train.images[mnist.train.labels < 5]
# y_train1 = mnist.train.labels[mnist.train.labels < 5]
# X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
# y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]
# X_test1 = mnist.test.images[mnist.test.labels < 5]
# y_test1 = mnist.test.labels[mnist.test.labels < 5]
#
#
# n_epochs = 1000
# batch_size = 20
#
# max_checks_without_progress = 20
# checks_without_progress = 0
# best_loss = np.infty
#
# with tf.Session() as sess:
#     init.run()
#
#     for epoch in range(n_epochs):
#         rnd_idx = np.random.permutation(len(X_train1))
#         for rnd_indices in np.array_split(rnd_idx, len(X_train1) // batch_size):
#             X_batch, y_batch = X_train1[rnd_indices], y_train1[rnd_indices]
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid1, y: y_valid1})
#         if loss_val < best_loss:
#             save_path = saver.save(sess, "./my_mnist_model_0_to_4.ckpt")
#             best_loss = loss_val
#             checks_without_progress = 0
#         else:
#             checks_without_progress += 1
#             if checks_without_progress > max_checks_without_progress:
#                 print("Early stopping!")
#                 break
#         print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
#             epoch, loss_val, best_loss, acc_val * 100))
#
# #Save the best model
# with tf.Session() as sess:
#     saver.restore(sess, "./my_mnist_model_0_to_4.ckpt")
#     acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
#     print("Final test accuracy: {:.2f}%".format(acc_test * 100))
#



#9.a solution

reset_graph()

#Restoring the learned model
restore_saver = tf.train.import_meta_graph("./my_best_mnist_model_0_to_4.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
loss = tf.get_default_graph().get_tensor_by_name("loss:0")
Y_proba = tf.get_default_graph().get_tensor_by_name("Y_proba:0")
logits = Y_proba.op.inputs[0]
accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")


#Exercise 9.a
learning_rate = 0.01

output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam2")
training_op = optimizer.minimize(loss, var_list=output_layer_vars)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
five_frozen_saver = tf.train.Saver()

#Making thte training, test and validation sets
X_train2_full = mnist.train.images[mnist.train.labels >= 5]
y_train2_full = mnist.train.labels[mnist.train.labels >= 5] - 5
X_valid2_full = mnist.validation.images[mnist.validation.labels >= 5]
y_valid2_full = mnist.validation.labels[mnist.validation.labels >= 5] - 5
X_test2 = mnist.test.images[mnist.test.labels >= 5]
y_test2 = mnist.test.labels[mnist.test.labels >= 5] - 5

#Sampling 100 examples per class
def sample_n_instances_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)

X_train2, y_train2 = sample_n_instances_per_class(X_train2_full, y_train2_full, n=100)
X_valid2, y_valid2 = sample_n_instances_per_class(X_valid2_full, y_valid2_full, n=30)

import time
print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_best_mnist_model_0_to_4")
    for var in output_layer_vars:
        var.initializer.run()

    t0 = time.time()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

    t1 = time.time()
    print("Total training time: {:.1f}s".format(t1 - t0))

# Checking accuracy with the training set
with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    Z = logits.eval(feed_dict={X: X_test2})
    y_pred = np.argmax(Z, axis=1)


# Code for displaying correctly classified and incorrectly classified
correctX = []
wrongX = []
count = 0
for i in range(y_pred.size):
    if y_test2[i] == 3:
        if y_pred[i] == 3:
            correctX.append(np.array([X_test2[i]]))
        else:
            wrongX.append(np.array([X_test2[i]]))
        count+=1
        if(count>99):
            break


import matplotlib.pyplot as plt
plt.figure(figsize=(3, 3 * len(wrongX)))
plt.subplot(121).set_label('correct')
plt.imshow(np.array(correctX)[:,0].reshape(28 * len(correctX), 28), cmap="binary", interpolation="nearest")
plt.axis('off')
plt.subplot(122).set_label('wrong')
plt.imshow(np.array(wrongX)[:,0].reshape(28 * len(wrongX), 28), cmap="binary", interpolation="nearest")
plt.axis('off')
plt.show()

