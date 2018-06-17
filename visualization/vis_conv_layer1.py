# -- coding: utf-8 --
"""
ASR visualize
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

chkp = "/tmp/speech_commands_train/conv.ckpt-36000"

tf.reset_default_graph()

filters = 64
v1 = tf.get_variable("Variable", shape=[20,8,1,filters])
v2 = tf.get_variable("Variable_2", shape=[10,4,filters,64])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, chkp)
    l1 = np.asarray(v1.eval())
    l2 = np.asarray(v2.eval())

    # Convolutional layer 1
    fig=plt.figure(figsize=(10, 10))

    for i in range(filters):
        fig.add_subplot(8, 8, i+1)
        plt.imshow(l1[:,:,0,i], cmap='gray')
    plt.show()

    fig=plt.figure(figsize=(10, 10))

    for i in range(filters):
        fig.add_subplot(8, 8, i+1)
        plt.imshow(l2[:,:,0,i], cmap='gray')
    plt.show()
