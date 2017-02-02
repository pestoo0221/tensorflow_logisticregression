#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Jidan Zhong 
# 2017- Feb- 1

import tensorflow as tf 
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def variable_summaries(var):                ###################################################################ADDDDDED########
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# LOAD DATA IRIS : 4 features, 3 groups (species), 50 samples for each spiece, so 150 samples together
iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]  # last sample removed
iris_y= pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

# numFeatures is the number of features in our input data.
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in.
numLabels = trainY.shape[1]

with tf.Graph().as_default():
    with tf.name_scope('input'):
        # 'None' means TensorFlow shouldn't expect a fixed number in that dimension
        X = tf.placeholder(tf.float32, [None, numFeatures], name='x-input') # Iris has 4 features, so X is a tensor to hold our data.
        yGold = tf.placeholder(tf.float32, [None, numLabels],name ='y-input') # This will be our correct answers matrix for 3 classes.
    
    with tf.name_scope('model'):
        with tf.name_scope('weights'):
            # W = tf.Variable(tf.zeros([numFeatures, numLabels], name='weight') # weight # 4-dimensional input and  3 classes
            weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights")) #Randomly sample from a normal distribution with standard deviation .01
            # tf.summary.scalar('weights', weights)
            variable_summaries(weights)
        with tf.name_scope('bias'):    
            # B = tf.Variable(tf.zeros([numLabels], name = 'bias') # bias # 3-dimensional output [0,0,1],[0,1,0],[1,0,0]
            bias = tf.Variable(tf.random_normal([1,numLabels], 
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))
            # tf.summary.scalar('bias', bias)
            variable_summaries(bias)
        with tf.name_scope('logistic_model'):
            # construct a model
            applyweight = tf.matmul(X, weights, name = 'applied_weights')
            addbias = tf.add(applyweight, bias, name = 'add_bias')
            y = tf.nn.sigmoid(addbias, name = 'activation')
            tf.summary.histogram('activations', y)

    with tf.name_scope('loss'):
        #Defining our cost function - Squared Mean Error
        loss = tf.nn.l2_loss(y-yGold, name="squared_error_cost")
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        # Defining our learning rate iterations (decay)
        learningRate = tf.train.exponential_decay(learning_rate=0.008,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)
        #Defining our Gradient Descent
        training = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yGold, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()    

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: # the first one means if the device doesnt exist, it can automatically appoint an existing device; 2nd means it will show the log infor for parameters and operations are on which device
        train_writer = tf.summary.FileWriter('/home/jidan/test/train', sess.graph)
        sess.run(init)

        # Initialize reporting variables
        convergenceTolerance = 0.0001
        cost = 0
        diff = 1
        epoch_values = []
        accuracy_values = []
        cost_values = []

        # Number of Epochs in our training
        numEpochs = 700
        # Training epochs
        for i in range(numEpochs):
            if i > 1 and diff < convergenceTolerance: # convergence tolerance
                print "change in cost %g; convergence."%(diff)
                break
            else:
                # Run training step
                yPred, _, acc, newCost, summary = sess.run([y,training,accuracy,loss, merged], feed_dict={X: trainX, yGold: trainY})
                train_writer.add_summary(summary, i)
                # Report occasional stats
                if i % 10 == 0:
                    # Add epoch to epoch_values
                    epoch_values.append(i)
                    # Add accuracy to live graphing variable
                    accuracy_values.append(acc)
                    # Add cost to live graphing variable
                    cost_values.append(newCost)
                    # Re-assign values for variables
                    diff = abs(newCost - cost)
                    cost = newCost

                    #generate print statements
                    print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, acc, newCost, diff))


        # How well do we perform on held-out test data?
        print("final accuracy on test set: %s" %str(sess.run(accuracy, feed_dict={X: testX, yGold: testY})))

plt.figure()
plt.subplot(211)
group0 = np.where(trainY[:,0] == 1)[0]
group1 = np.where(trainY[:,1] == 1)[0]
group2 = np.where(trainY[:,2] == 1)[0]
g0, = plt.plot(trainX[group0,0], trainX[group0,1], 'ro')
g1, = plt.plot(trainX[group1,0], trainX[group1,1], 'bo')
g2, = plt.plot(trainX[group2,0], trainX[group2,1], 'go')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Original Data")
plt.legend([g0, g1, g2], ["Setosa", "Versicolor", "Virginica"])
# legend "Setosa"  "Versicolor"  "Virginica"
# plt.show() 
plt.subplot(212)
group0p = np.where(np.argmax(yPred, 1)==0)[0]
group1p = np.where(np.argmax(yPred, 1)==1)[0]
group2p = np.where(np.argmax(yPred, 1)==2)[0] 
g0, = plt.plot(trainX[group0p,0], trainX[group0p,1], 'ro')
g1, = plt.plot(trainX[group1p,0], trainX[group1p,1], 'bo')
g2, = plt.plot(trainX[group2p,0], trainX[group2p,1], 'go')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("groups based on classification")
plt.legend([g0, g1, g2], ["Setosa", "Versicolor", "Virginica"])
# legend "Setosa"  "Versicolor"  "Virginica"
plt.show() 
