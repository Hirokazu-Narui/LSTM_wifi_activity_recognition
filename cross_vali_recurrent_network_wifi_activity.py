'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import sklearn as sk
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.python.ops import rnn, rnn_cell
from sklearn.model_selection import KFold, cross_val_score
import csv


# Import WiFi Activity data
from cross_vali_input_data import csv_import, DataSet
window_size = 50 #50 sampling (Data sampling rate = 50 Hz, so it means 1 sec)
threshold = 60 #more than 60% sampling is recognized activity

# Parameters
input_data_type = 1 #1:Amplitude only, 2: Phase only, 3: Amplitude + Phase
learning_rate = 0.0001 #original 0.0001
training_iters = 2000 #original 2000
batch_size = 500 #original 500
display_step = 100

# Network Parameters

# WiFi activity data input ( Amplitude:90, Phase:90, Amplitude+Phase:180 )
if input_data_type == 1:
	n_input = 90
	tmp = "Amp"

elif input_data_type == 2:
	n_input = 90
	tmp = "Phase"

else:
	n_input = 180
	tmp = "Amp+Phase"

n_steps = window_size # timesteps
n_hidden = 200 # hidden layer # of features original 200
n_classes = 7 # WiFi activity total classes ( bed, fall, walk, pickup, run, sitdown, standup )

# Save folder name
foldername = tmp + "_lr" + str(learning_rate) + "_bs" + str(batch_size) + "_nh" + str(n_hidden)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

	# Permuting batch_size and n_steps
	x = tf.transpose(x, [1, 0, 2])
	# Reshaping to (n_steps*batch_size, n_input)
	x = tf.reshape(x, [-1, n_input])
	# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	x = tf.split(0, n_steps, x)

	# Define a lstm cell with tensorflow
	lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

	# Get lstm cell output
	outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']


##### main #####
pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
cvscores = []
confusion_sum = [[0 for i in range(7)] for j in range(7)]

#data import
x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk, \
y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk = csv_import( input_data_type  )

print(" bed =",len(x_bed), " fall=", len(x_fall), " pickup =", len(x_pickup), " run=", len(x_run), " sitdown=", len(x_sitdown), " standup=", len(x_standup), " walk=", len(x_walk))

kk = 10 # k_fold

## Main ##
# mkdir + chdir
os.mkdir(foldername)
os.chdir(foldername)

with tf.Session() as sess:
	for i in range(kk):

		#Initialization
		train_loss = []
		train_acc = []
		validation_loss = []
		validation_acc = []

		#Roll the data
		x_bed = np.roll(x_bed, int(len(x_bed) / kk), axis=0)
		y_bed = np.roll(y_bed, int(len(y_bed) / kk), axis=0)
                x_fall = np.roll(x_fall, int(len(x_fall) / kk), axis=0)
                y_fall = np.roll(y_fall, int(len(y_fall) / kk), axis=0)
                x_pickup = np.roll(x_pickup, int(len(x_pickup) / kk), axis=0)
                y_pickup = np.roll(y_pickup, int(len(y_pickup) / kk), axis=0)
                x_run = np.roll(x_run, int(len(x_run) / kk), axis=0)
                y_run = np.roll(y_run, int(len(y_run) / kk), axis=0)
                x_sitdown = np.roll(x_sitdown, int(len(x_sitdown) / kk), axis=0)
                y_sitdown = np.roll(y_sitdown, int(len(y_sitdown) / kk), axis=0)
                x_standup = np.roll(x_standup, int(len(x_standup) / kk), axis=0)
                y_standup = np.roll(y_standup, int(len(y_standup) / kk), axis=0)
                x_walk = np.roll(x_walk, int(len(x_walk) / kk), axis=0)
                y_walk = np.roll(y_walk, int(len(y_walk) / kk), axis=0)

		#data separation
		wifi_x_train = np.r_[x_bed[int(len(x_bed) / kk):], x_fall[int(len(x_fall) / kk):], x_pickup[int(len(x_pickup) / kk):], \
					x_run[int(len(x_run) / kk):], x_sitdown[int(len(x_sitdown) / kk):], x_standup[int(len(x_standup) / kk):], x_walk[int(len(x_walk) / kk):]]

		wifi_y_train = np.r_[y_bed[int(len(y_bed) / kk):], y_fall[int(len(y_fall) / kk):], y_pickup[int(len(y_pickup) / kk):], \
                        		y_run[int(len(y_run) / kk):], y_sitdown[int(len(y_sitdown) / kk):], y_standup[int(len(y_standup) / kk):], y_walk[int(len(y_walk) / kk):]]

		wifi_y_train = wifi_y_train[:,1:]

		wifi_x_validation = np.r_[x_bed[:int(len(x_bed) / kk)], x_fall[:int(len(x_fall) / kk)], x_pickup[:int(len(x_pickup) / kk)], \
                        		x_run[:int(len(x_run) / kk)], x_sitdown[:int(len(x_sitdown) / kk)], x_standup[:int(len(x_standup) / kk)], x_walk[:int(len(x_walk) / kk)]]

		wifi_y_validation = np.r_[y_bed[:int(len(y_bed) / kk)], y_fall[:int(len(y_fall) / kk)], y_pickup[:int(len(y_pickup) / kk)], \
                        		y_run[:int(len(y_run) / kk)], y_sitdown[:int(len(y_sitdown) / kk)], y_standup[:int(len(y_standup) / kk)], y_walk[:int(len(y_walk) / kk)]]

		wifi_y_validation = wifi_y_validation[:,1:]

		#data set
		wifi_train = DataSet(wifi_x_train, wifi_y_train)
		wifi_validation = DataSet(wifi_x_validation, wifi_y_validation)
		print(wifi_x_train.shape, wifi_y_train.shape, wifi_x_validation.shape, wifi_y_validation.shape)
		saver = tf.train.Saver()
		sess.run(init)
		step = 1

		# Keep training until reach max iterations
		while step < training_iters:
			batch_x, batch_y = wifi_train.next_batch(batch_size)
			x_vali = wifi_validation.images[:]
			y_vali = wifi_validation.labels[:]
			# Reshape data
			batch_x = batch_x.reshape((batch_size, n_steps, n_input))
			x_vali = x_vali.reshape((-1, n_steps, n_input))
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			acc_vali = sess.run(accuracy, feed_dict={x: x_vali, y: y_vali})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			loss_vali = sess.run(cost, feed_dict={x: x_vali, y: y_vali})

			# Store the accuracy and loss
			train_acc.append(acc)
			train_loss.append(loss)
			validation_acc.append(acc_vali)
			validation_loss.append(loss_vali)
 
			if step % display_step == 0:
				print("Iter " + str(step) + ", Minibatch Training  Loss= " + \
					"{:.6f}".format(loss) + ", Training Accuracy= " + \
					"{:.5f}".format(acc) + ", Minibatch Validation  Loss= " + \
					"{:.6f}".format(loss_vali) + ", Validation Accuracy= " + \
					"{:.5f}".format(acc_vali) )
			step += 1

		#Calculate the confusion_matrix
		cvscores.append(acc_vali * 100)
		y_p = tf.argmax(pred, 1)
		val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_vali, y: y_vali})
		y_true = np.argmax(y_vali,1)
		print(sk.metrics.confusion_matrix(y_true, y_pred))
		confusion = sk.metrics.confusion_matrix(y_true, y_pred)
		confusion_sum = confusion_sum + confusion

		#Save the Accuracy curve
		fig = plt.figure(2 * i - 1)
		plt.plot(train_acc)
		plt.plot(validation_acc)
		plt.xlabel("n_epoch")
		plt.ylabel("Accuracy")
		plt.legend(["train_acc","validation_acc"],loc=4)
		plt.ylim([0,1])
		plt.savefig(("Accuracy_" + str(i) + ".png"), dpi=150)

		#Save the Loss curve
		fig = plt.figure(2 * i)
		plt.plot(train_loss)
		plt.plot(validation_loss)
		plt.xlabel("n_epoch")
		plt.ylabel("Loss")
		plt.legend(["train_loss","validation_loss"],loc=1)
		plt.ylim([0,2])
		plt.savefig(("Loss_" + str(i) + ".png"), dpi=150)

		#Check the input data
#		wifi_x_train_chk = wifi_x_train.reshape(len(wifi_x_train), -1)
#		wifi_x_validation_chk = wifi_x_validation.reshape(len(wifi_x_validation), -1)
#		with open("wifi_x_train_" + str(i) + "fold.csv", "w") as f:
#			writer = csv.writer(f, lineterminator="\n")
#			writer.writerows(wifi_x_train_chk)
#		with open("wifi_x_vaidation_" + str(i) + "fold.csv", "w") as f:
#                        writer = csv.writer(f, lineterminator="\n")
#                        writer.writerows(wifi_x_validation_chk)
#		with open("wifi_y_train_" + str(i) + "fold.csv", "w") as f:
#			writer = csv.writer(f, lineterminator="\n")
#			writer.writerows(wifi_y_train)
#		with open("wifi_y_validation_" + str(i) + "fold.csv", "w") as f:
#			writer = csv.writer(f, lineterminator="\n")
#			writer.writerows(wifi_y_validation)

	print("Optimization Finished!")
	print("%.1f%% (+/- %.1f%%)" % (np.mean(cvscores), np.std(cvscores)))
	saver.save(sess, "model.ckpt")

	#Save the confusion_matrix
	np.savetxt("confusion_matrix.txt", confusion_sum, delimiter=",", fmt='%d')
	np.savetxt("accuracy.txt", (np.mean(cvscores), np.std(cvscores)), delimiter=".", fmt='%.1f')
