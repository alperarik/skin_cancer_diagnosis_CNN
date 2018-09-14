import tensorflow as tf
import numpy as np
import os
from scipy import misc # feel free to use another image loader
import time
from functools import reduce
start_time = time.time()

# import data
list_of_images = [f for f in os.listdir('./Dataset500') if (f.endswith('.jpg'))]
list_of_images.sort()
list_of_labels = [line for line in open('train.csv')]
list_of_labels.sort()
list_of_labels = [line.split(',')[1] for line in list_of_labels]
list_of_labels = [line.replace('\n','') for line in list_of_labels]

set_of_labels = set(list_of_labels)
set_of_labels = list(set_of_labels)

def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

# Global variables
IMAGE_SIZE = 500
LABEL_COUNT = len(list_of_labels)
LABEL_SIZE = len(set_of_labels)
BATCH_SIZE = 10
NUM_OF_EPOCHS = 1000

newList = []
for i in list_of_labels:
    for j in set_of_labels:
        if (i == j):
            tmp = set_of_labels.index(i)
            tmpList = [0]*LABEL_SIZE
            tmpList[tmp] = 1
            newList.append(tmpList)

list_of_labels = newList

def create_batches():
  images = []
  for img in list_of_images:
    images.append(misc.imread('./Dataset500/'+img))
  images = np.asarray(images)
  labels = np.asarray(list_of_labels)
  #print(images.shape)
  #print(labels.shape)
  while (True):
    for i in range(0,LABEL_COUNT,BATCH_SIZE):
      yield(images[i:i+BATCH_SIZE],labels[i:i+BATCH_SIZE])


# Create the model
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE * 3])
y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])
#W = tf.Variable(tf.zeros([IMAGE_SIZE * IMAGE_SIZE * 3, LABEL_SIZE]))
#b = tf.Variable(tf.zeros([LABEL_SIZE]))
W = tf.Variable(tf.random_normal([IMAGE_SIZE * IMAGE_SIZE * 3, LABEL_SIZE], stddev=0.2, mean=128))
b = tf.Variable(tf.random_normal([LABEL_SIZE], stddev=0.2, mean=128))
y = tf.nn.softmax(tf.matmul(x, W) + b)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')

#FOR weight_variable
#[A,B,C,D], A and B input size, C is input dimension, D is output dimension
W_conv1 = weight_variable([32, 32, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([32, 32, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7* 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 *7 * 64])  #convert vector
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #make dropout

W_fc2 = weight_variable([1024, LABEL_SIZE])
b_fc2 = bias_variable([LABEL_SIZE])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
#cross_entropy = -tf.reduce_sum( (  (y_*tf.log(y_conv + 1e-9)) + ((1-y_) * tf.log(1 - y_conv + 1e-9)) ))
cross_entropy = -tf.reduce_mean( (  (y_*tf.log(y_conv + 1e-9)) + ((1-y_) * tf.log(1 - y_conv + 1e-9)) ))
#train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

batch_generator = create_batches()
for i in range(NUM_OF_EPOCHS):
    if i % 100 == 0:
        x_batch, y_batch = batch_generator.__next__()
        #print(x_batch.shape)
        #print(y_batch.shape)
        x_batch = np.reshape(x_batch, (-1,IMAGE_SIZE*IMAGE_SIZE*3))
        y_batch = np.reshape(y_batch, (-1,LABEL_SIZE))
        print(x_batch.shape)
        print(y_batch.shape)
        train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.75})  # %25 dropout

save_path = saver.save(sess, "opusModel.ckpt")  # saving model
print ("Model saved in file: ", save_path)
print(secondsToStr(time.time() - start_time))

start_time = time.time()

print('TEST ACCURACY')
BATCH_SIZE = LABEL_COUNT / 3 # %33 percentage split testing :)
batch_generator = create_batches()
x_batch, y_batch = batch_generator.__next__()
x_batch = np.reshape(x_batch, (-1, IMAGE_SIZE * IMAGE_SIZE * 3))
y_batch = np.reshape(y_batch, (-1, LABEL_SIZE))
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: x_batch, y_: y_batch, keep_prob: 1.0}))
print(secondsToStr(time.time() - start_time))
sess.close()