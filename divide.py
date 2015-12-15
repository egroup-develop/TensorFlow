#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
from getName import getName
from getName import getIndex
from PIL import Image


NUM_CLASSES = 245
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('use_model', 'model_245_1470_490_8per.ckpt', 'File name of model data.')

####################モデルを作成する関数#####################
# 引数: 画像のplaceholder, dropout率のplace_holder
# 返り値: モデルの計算結果
#############################################################
def inference(images_placeholder, keep_prob):
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

if __name__ == '__main__':
    test_image = []
    name = sys.argv
    for i in range(1, len(sys.argv)):
        img = cv2.imread(sys.argv[i])
        img = cv2.resize(img, (28, 28))
        test_image.append(img.flatten().astype(np.float32)/255.0)
    test_image = np.asarray(test_image)

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    #分類器として使うmodelを指定する
    #saver.restore(sess, FLAGS.use_model)
    saver.restore(sess, "model_245_1470_490_8per.ckpt")

    ##### ランクづけ #####
    for i in range(len(test_image)):
        pred = np.argmax(logits.eval(feed_dict={ 
            images_placeholder: [test_image[i]],
            keep_prob: 1.0 })[0])
        print str(name[i + 1]) + "は" + str(pred) + " = " + str(getName(pred)) + " さんに最も近いです"

        # 特徴の近さ確率をノード数分
        feature = logits.eval(feed_dict={
            images_placeholder: [test_image[i]],
            keep_prob: 1.0 })[0]

        featureSorted = feature
        featureSorted = np.sort(featureSorted)
        featureSorted = featureSorted[-1::-1]
        pred2 = featureSorted[1]
        pred3 = featureSorted[2]
        pred4 = featureSorted[3]
        pred5 = featureSorted[4]
        pred6 = featureSorted[5]

        print "つまり" + str(getName(pred)) + "さんは, "

        rank = {}
        for index, value in enumerate(feature):
          if pred2 == value:
            rank["1"] = str(getName(index))
            print "2番目に" + str(index) + " = " + str(getName(index)) + " さんに近いです"
          elif pred3 == value:
            rank["2"] = str(getName(index))
            print "3番目に" + str(index) + " = " + str(getName(index)) + " さんに近いです"
          elif pred4 == value:
            rank["3"] = str(getName(index))
            print "4番目に" + str(index) + " = " + str(getName(index)) + " さんに近いです"
          elif pred5 == value:
            rank["4"] = str(getName(index))
            print "5番目に" + str(index) + " = " + str(getName(index)) + " さんに近いです"
          elif pred6 == value:
            rank["5"] = str(getName(index))
            print "6番目に" + str(index) + " = " + str(getName(index)) + " さんに近いです"
        print "\n"

        ##### 入力した画像の分類結果(クラス)から名前と画像を取得 #####
        rank =  sorted(rank.items(), key=lambda x:x[0])
        for i in range(len(rank)):
          print rank[i][1]
          print getIndex(rank[i][1])

          for j in range(4):
            imagePath = "LogirlImages/" + getIndex(rank[i][1]) + "/" + "image_" + str(j+1) + "_origin.jpeg"
            print imagePath
            ##### 取得した画像を表示 #####
            image = Image.open(imagePath)
            image.show()
