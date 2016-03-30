#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modelの構築はinference, loss, trainingの3つのサブルーチンに分けるのが普通
# http://www.tensorflow.org/tutorials/mnist/tf/index.html
# 本処理は下記2ファイルの統合版
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/tutorials/mnist/mnist.py
#  2015/12/08現在: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist
# MINIST for Expertsでは28x28x1->28x28x32->14x14x32->14x14x64->7x7x64->1024->10
# 参考: http://qiita.com/supersaiakujin/items/bc05b9f329aca48329ac

import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import datetime
import time

# 多値分類ということで
NUM_CLASSES = 328
# 画像の縦/横画素数(S)
IMAGE_SIZE = 28
# SxSxNでNは枚数. 入力画像がグレースケールならN=1, カラーならRGB計3枚でN=3
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
dateTime = datetime.datetime.today()

# flagsを使ってmodelで使用する定数をハンドリングする
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'DailyLogirlDateSet_1to328_28px/train.txt', 'File name of train data')
flags.DEFINE_string('test', 'DailyLogirlDateSet_1to328_28px/test.txt', 'File name of test data')
# tensorboardに出力するファイルの置き場
flags.DEFINE_string('train_dir', './data/', 'Directory to put the training data.')
flags.DEFINE_string('save_model', './model/model_327_28px_' + str(dateTime.month) + str(dateTime.day) + str(dateTime.hour) + str(dateTime.minute) + '.ckpt', 'File name of model data.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
# 学習率は0.01
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

################ inference ####################
# 予測モデルを作成する関数. NNの構想
# 引数: 画像のplaceholder, dropout率のplaceholder
# 返り値: 各クラスの確率のようなもの(出力の予測を含むTensor)
# 入力: 28x28x3(縦画素x横画素x枚数)
# 畳み込み: 2回
# プーリング: 2回
# 内積(全結合層)->ソフトマックス関数->Loss出力
###############################################
def inference(images_placeholder, keep_prob):
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 畳み込み層の作成(nn.conv2dの2dは2-D convolutionの意)
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成(プーリングは3種用意されている)
    def max_pool(name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    # 正規化
    def norm(name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    # 入力画像(Tensor)を28*28*3にリシェイプ
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    ##########畳み込み層##########
    # 畳み込み
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([3, 3, 3, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Max pooling(down-sampling)
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool('pool1', h_conv1, k=2)

    # Normalization
    with tf.name_scope('norm1') as scope:
        h_norm1 = norm('norm1', h_pool1, lsize=4)

    # Dropout
    h_norm1 = tf.nn.dropout(h_norm1, keep_prob)

    ##########畳み込み層##########
    # 畳み込み
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([3, 3, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)

    # Max pooling(down-sampling)
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool('pool2', h_conv2, k=2)

    # Normalization
    with tf.name_scope('norm2') as scope:
        h_norm2 = norm('norm2', h_pool2, lsize=4)

    # Dropout
    h_norm2 = tf.nn.dropout(h_norm2, keep_prob)

    ##########畳み込み層##########
    # 畳み込み
    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 128, 256])
        b_conv3 = bias_variable([256])
        h_conv3 = tf.nn.relu(conv2d(h_norm2, W_conv3) + b_conv3)

    # Max pooling(down-sampling)
    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool('pool3', h_conv3, k=2)

    # Normalization
    with tf.name_scope('norm3') as scope:
        h_norm3 = norm('norm3', h_pool3, lsize=4)

    # Dropout
    h_norm3 = tf.nn.dropout(h_norm3, keep_prob)

    ###########全結合層##########
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([4*4*256, 1024]) # 7*7*64次元の画像のベクトルを1024のベクトルにする
        b_fc1 = bias_variable([1024])
        h_norm3_flat = tf.reshape(h_norm3, [-1, 4*4*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_norm3_flat, W_fc1) + b_fc1) # matmul: 行列積Matrix Multiply

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, 1024]) # 7*7*64次元の画像のベクトルを1024のベクトルにする
        b_fc2 = bias_variable([1024])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2) # matmul: 行列積Matrix Multiply

    # TensorBoardでひとかたまりのノードとして表示される
    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([1024, NUM_CLASSES]) # 1024次元ベクトルを目的のクラス数分にする
        b_fc3 = bias_variable([NUM_CLASSES])

    ###########出力層##########
    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

    return tf.clip_by_value(y_conv, 1e-10, 1.0)

####################### loss ##########################
# loss関数
# inference()から得た予測値からバックプロパゲーションに使う損失関数(誤差)を計算
# 引数: ロジットのtensor(float - [batch_size, NUM_CLASSES] = y_conv), ラベルのtensor(int32 - [batch_size, NUM_CLASSES])
# 返り値: 交差エントロピーのtensor, float 
#######################################################
def loss(logits, labels):
    # 交差エントロピーの計算
    # tf.reduce_sum: 全てのテンソルを加算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))

    # TensorBoardで表示するよう指定
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

######################### training ####################
# 学習の実行
# loss()で得た誤差を逆伝搬してネットワークを学習させる. 学習のop(node)を定義する
#  https://www.tensorflow.org/versions/master/api_docs/python/train.html#optimizers
# 引数: 損失のTensor(loss()の結果cross_entropy), 学習係数
# 返り値: 学習のために定義したop
#######################################################
def training(loss, learning_rate):
    # AdamOptimizerは全体のネットを最適化(自動微分のような)してくれる関数. 要は学習率l_rで最適化アルゴリズムを使うことで誤差を最小化する
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step
#    # Create the gradient descent optimizer with the given learning rate.
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#    # Create a variable to track the global step.
#    global_step = tf.Variable(0, name='global_step', trainable=False)
#    # Use the optimizer to apply the gradients that minimize the loss
#    # (and also increment the global step counter) as a single training step.
#    train_op = optimizer.minimize(loss, global_step=global_step)
#    return train_op

############################ 正解率の計算(モデルの評価) ######################
# 正解率(accuracy)を計算する関数
# 引数: inference()の結果y_conv, ラベルのTensor(int32 - [batch_size, NUM_CLASSES])
# 返り値: 正解率(float)
################################################################
def accuracy(logits, labels):
    # モデルのクラスラベルと正解のクラスラベルが等しければ正解(予測は正しい)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # booleanリストを浮動小数点型にキャスト(True->1.0, False->0.0)し, その平均値をとる
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 正解率をsummaryし, tf.train.SummaryWriterでそれを書き出す
    tf.scalar_summary("accuracy", accuracy)
    return accuracy

############### データセットの読み込み・学習の実行 #############
# TensorFlowにはdecode_pngのような(jpegも然り)画像を読み込むための関数が用意されている
# http://www.tensorflow.org/api_docs/python/image.html#decode_png
# 今回はこれを使わずcv2で読み込む
################################################################
if __name__ == '__main__':
    # 実行時間を測る
    start = time.time()

    # ファイルを開く
    with open(FLAGS.train, 'r') as f:
        # データを入れる配列
        train_image = []
        train_label = []
        for line in f:
            # 文字列の末尾(改行)を除いてスペース区切りにする
            line = line.rstrip()
            l = line.split()
            # データを読み込んで28x28に縮小
            img = cv2.imread(l[0])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            # 一列にした後、0-1のfloat値にする
            train_image.append(img.flatten().astype(np.float32)/255.0)
            # ラベルを1-of-n方式で用意する
            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])] = 1
            train_label.append(tmp)
        # numpy形式に変換
        train_image = np.asarray(train_image)
        train_label = np.asarray(train_label)
        train_len = len(train_image)

    with open(FLAGS.test, 'r') as f:
        test_image = []
        test_label = []
        for line in f:
            line = line.rstrip()
            l = line.split()
            img = cv2.imread(l[0])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            test_image.append(img.flatten().astype(np.float32)/255.0)
            tmp = np.zeros(NUM_CLASSES)
            tmp[int(l[1])] = 1
            test_label.append(tmp)
        test_image = np.asarray(test_image)
        test_label = np.asarray(test_label)
        test_len = len(test_image)

    # VariableをGraphに追加するにはwith tf.Graph().as_default():スコープ内で宣言もしくは呼び出す必要がある
    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        images_placeholder = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
        # クラスラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder(tf.float32, [None, NUM_CLASSES])
        # dropout率を入れる仮のTensor. dropout(keep probability)
        keep_prob = tf.placeholder(tf.float32)

        # inference()を呼び出してモデルを作る
        logits = inference(images_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = training(loss_value, FLAGS.learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_placeholder)

        # 保存の準備. Create a saver for writing training checkpoints
        saver = tf.train.Saver()
        # Sessionの作成. Create a session for running Ops on the Graph
        sess = tf.Session()
        # 変数の初期化
        sess.run(tf.initialize_all_variables())
        # TensorBoardで表示する値の設定
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        # 訓練の実行
        if train_len % FLAGS.batch_size is 0:
            train_batch = train_len/FLAGS.batch_size
        else:
            train_batch = (train_len/FLAGS.batch_size)+1
        for step in range(FLAGS.max_steps):
            for i in range(train_batch):
                # batch_size分の画像に対して訓練の実行
                batch = FLAGS.batch_size*i
                batch_plus = FLAGS.batch_size*(i+1)
                if batch_plus > train_len: batch_plus = train_len
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                  images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                  labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                  keep_prob: 0.5})

            if step % 10 == 0:
                # 10 step終わるたびに精度を計算する
                train_accuracy = 0.0
                for i in range(train_batch):
                    batch = FLAGS.batch_size*i
                    batch_plus = FLAGS.batch_size*(i+1)
                    if batch_plus > train_len: batch_plus = train_len
                    train_accuracy += sess.run(acc, feed_dict={
                        images_placeholder: train_image[batch:batch_plus],
                        labels_placeholder: train_label[batch:batch_plus],
                        keep_prob: 1.0})
                    if i is not 0: train_accuracy /= 2.0
                # 10 step終わるたびにTensorBoardに表示する値を追加する
                #summary_str = sess.run(summary_op, feed_dict={
                #    images_placeholder: train_image,
                #    labels_placeholder: train_label,
                #    keep_prob: 1.0})
                #summary_writer.add_summary(summary_str, step)
                print "step %d, training accuracy %g"%(step, train_accuracy)

    # 訓練が終了したらテストデータに対する精度を表示
    print "train finish!!\n\n\ntest start."
    if test_len % FLAGS.batch_size is 0:
        test_batch = test_len/FLAGS.batch_size
    else:
        test_batch = (test_len/FLAGS.batch_size)+1
        print "test_batch = "+str(test_batch)
    test_accuracy = 0.0
    for i in range(test_batch):
        batch = FLAGS.batch_size*i
        batch_plus = FLAGS.batch_size*(i+1)
        if batch_plus > train_len: batch_plus = train_len
        test_accuracy += sess.run(acc, feed_dict={
                images_placeholder: test_image[batch:batch_plus],
                labels_placeholder: test_label[batch:batch_plus],
                keep_prob: 1.0})
        if i is not 0: test_accuracy /= 2.0
    print "test accuracy %g"%(test_accuracy)

    # 訓練が終了したらテストデータに対する精度を表示. 対評価データ
    print "test accuracy %g"%sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0})

    # 実行時間を表示
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

    # 最終的なモデルを保存
    save_path = saver.save(sess, FLAGS.save_model)
