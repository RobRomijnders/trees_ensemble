# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: rob
"""
import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()


#Load the data
data_train = genfromtxt('ModelingTrain.csv',delimiter=',',skip_header=1)
data_test = genfromtxt('ModelingTest.csv',delimiter=',',skip_header=1)
N = data_train.shape[0]

#Create validation set
Nval = 7000
ind = np.random.permutation(N)

X_train = data_train[ind[:7000],2:]
y_train = data_train[ind[:7000],1]

X_val = data_train[ind[7000:],2:]
y_val = data_train[ind[7000:],1]


N,D = X_train.shape
Nval = X_val.shape[0]

#Check for the input sizes
assert (N>D), 'You are feeding a fat matrix for training, are you sure?'
assert (Nval>D), 'You are feeding a fat matrix for testing, are you sure?'

# Nodes for the input variables
x = tf.placeholder("float", shape=[None, D], name = 'Input_data')
y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')


# Define functions for initializing variables and standard layers
#For now, this seems superfluous, but in extending the code
#to many more layers, this will keep our code
#read-able
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

#with tf.name_scope("Conv1") as scope:
#  W_conv1 = weight_variable([5, 5, 1, 32], 'Conv_Layer_1')
#  b_conv1 = bias_variable([32], 'bias_for_Conv_Layer_1')
#  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#  h_pool1 = max_pool_2x2(h_conv1)
#  
## The name_scope lines serve to organize our graphs that TensorFlow will create
## for us
#with tf.name_scope("Conv2") as scope:
#  W_conv2 = weight_variable([5, 5, 32, 64], 'Conv_Layer_2')
#  b_conv2 = bias_variable([64], 'bias_for_Conv_Layer_2')
#  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#  h_pool2 = max_pool_2x2(h_conv2)
h1 = 50
h2 = 12

with tf.name_scope("Fully_Connected1") as scope:
  W_fc1 = weight_variable([D, h1], 'Fully_Connected_layer_1')
  b_fc1 = bias_variable([h1], 'bias_for_Fully_Connected_Layer_1')
  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

with tf.name_scope("Fully_Connected2") as scope:
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  W_fc2 = weight_variable([h1, h2], 'Fully_Connected_layer_2')
  b_fc2 = bias_variable([h2], 'bias_for_Fully_Connected_Layer_2')
  h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  
with tf.name_scope("Fully_Connected3") as scope:
  h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
  
  W_fc3 = weight_variable([h2, 7], 'Fully_Connected_layer_3')
  b_fc3 = bias_variable([7], 'bias_for_Fully_Connected_Layer_3')
  h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

# Also add histograms to TensorBoard
w1_hist = tf.histogram_summary("W_fc1", W_fc1)
b1_hist = tf.histogram_summary("b_fc1", b_fc1)
w2_hist = tf.histogram_summary("W_fc2", W_fc2)
b2_hist = tf.histogram_summary("b_fc2", b_fc2)

with tf.name_scope("Softmax") as scope:
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc3, y_, name='Cross_entropy_loss')
    loss = tf.reduce_sum(ce_loss)
    ce_summ = tf.scalar_summary("cross entropy", loss)
with tf.name_scope("train") as scope:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
with tf.name_scope("Evaluating") as scope:
    correct_prediction = tf.equal(tf.argmax(h_fc2,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)
    
    
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/home/rob/Dropbox/DataMining/assignment2/log_tb", sess.graph_def)

#Collect the accuracies in a numpy array
iterations = 5000
acc_collect = np.zeros((2,iterations//100))
step = 0

sess.run(tf.initialize_all_variables())
for i in range(iterations):
  batch_ind = np.random.choice(N,100,replace=False)
  if i%100 == 0:
    result = sess.run([accuracy,merged], feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0})
    acc = result[0]
    acc_collect[0,step] = acc
    summary_str = result[1]
    writer.add_summary(summary_str, i)
    writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
    print("Accuracy at step %s: %s" % (i, acc))
    #Now also obtain the train_accuracy
    result = sess.run([accuracy,merged], feed_dict={ x: X_train, y_: y_train, keep_prob: 1.0})
    acc_collect[1,step] = result[0]
    step+=1
  sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 0.5})
plt.plot(acc_collect[0],label='Valid')
plt.plot(acc_collect[1],label = 'Train')
plt.legend()
plt.show()


 
sess.close()

# We can now open TensorBoard. Run the following line from your terminal
# tensorboard --logdir=/home/rob/Dropbox/ConvNets/tf/log_tb

#With one fc layer of 50 neurons, we go to 94% val accuracy