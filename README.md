# 実験結果
- 第1回
  - データセット: デ**ーロ**ルの100人. 1人につき4枚
    - 訓練: 300枚
    - 評価: 100枚
    - 正規化: していない(顔切り抜きスクリプトは作った. 最終確認は目視で, 単にそれが大変なので後回しにした)
  - ニューラルネット
    - 学習率: 0.01
    - 畳み込み 2回
    - 活性化関数ReLU 2回
    - プーリング 2回
    - fc1層はドロップアウトさせている
    - fc2層の値を取り出してソフトマックス関数にて正規化->確率 = クラス
    - 入力は28x28x3
    - ステップ数は200
    - 学習
      - 要した時間: 約15分
      - ログ: 

```
...
step 196, training accuracy 1
step 197, training accuracy 1
step 198, training accuracy 1
step 199, training accuracy 1
test accuracy 0.02
```

      - 正解率: 2 %
      - ![正解率とloss率のグラフ](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/1_グラフ.png)
      - ![NNグラフ](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/1_NNグラフ.png)
  - 生成した分類器で画像の分類を行ってみた
    - 訓練データは全部当ててくれた
    - 評価データは10回ほどしか試していないが当たりにあたっていない..2パー...
  - 所感
    - まあNNを組む練習なので動いたし当然だけど訓練データはしっかり正解しているようなのでよし
    - データセットを超増やして正規化もすればよい分類器ができる予感
    - 同じデータセットでの学習だったがcaffeより3倍くらい早かったかも

- 第2回
  - データセット: デ**ーロ**ルの100人. 1人につき8枚
    - 訓練: 600枚
    - 評価: 200枚
    - 正規化: ImageMagickで行った
  - ニューラルネット
    - 学習率: 0.01
    - 畳み込み 2回
    - 活性化関数ReLU 2回
    - プーリング 2回
    - fc1層はドロップアウトさせている
    - fc2層の値を取り出してソフトマックス関数にて正規化->確率 = クラス
    - 入力は28x28x3
    - ステップ数は200
    - 学習
      - 要した時間: 約40分
      - ログ:
        - 
...
step 197, training accuracy 1
step 198, training accuracy 1
step 199, training accuracy 1
test accuracy 0.16
      - 正解率: 16 %
- 第3回
  - データセット
    - 第2回同様
  - ニューラルネット
    - 学習率: 1*e^-6(前回の約1/10. 過学習対策)
      - 学習率を低くすると, 本来学習してほしい特徴以外の訓練データにしか存在しない特徴を学習しにくくなるため, 未知のデータに対して正解しやすくなる. その分学習にかかる正解率は緩やかな上昇となり時間がかかる
    - 他は第2回同様
    - ログ: 
      - step 197, training accuracy 0.0583333
step 198, training accuracy 0.06
step 199, training accuracy 0.0616667
test accuracy 0.03 
    - 正解率: 3%
- 第4回
  - データセット: デ**ーロ**ルの245人. 1人につき8枚
    - 訓練: 1470枚
    - 評価: 490枚
    - 他は第2回同様
  - ニューラルネット
    - 学習率: 0.01
    - 他は第2回同様
    - ログ:
      - 

```
...
step 151, training accuracy 1
step 152, training accuracy 1
step 153, training accuracy 1
step 154, training accuracy 1
W tensorflow/core/common_runtime/executor.cc:1027] 0x7ff0dd4efd90 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv1/add)]]
W tensorflow/core/common_runtime/executor.cc:1027] 0x7ff0dd4efd90 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv2/Relu_grad/conv2/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv2/add)]]
W tensorflow/core/common_runtime/executor.cc:1027] 0x7ff0dd4efd90 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/fc1/Relu_grad/fc1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](fc1/add)]]
Traceback (most recent call last):
  File "featureProto.py", line 226, in <module>
    keep_prob: 0.5})
  File "/Library/Python/2.7/site-packages/tensorflow/python/client/session.py", line 345, in run
    results = self._do_run(target_list, unique_fetch_targets, feed_dict_string)
  File "/Library/Python/2.7/site-packages/tensorflow/python/client/session.py", line 419, in _do_run
    e.code)
tensorflow.python.framework.errors.InvalidArgumentError: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv1/add)]]
Caused by op u'gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics', defined at:
  File "featureProto.py", line 203, in <module>
    train_op = training(loss_value, FLAGS.learning_rate)
  File "featureProto.py", line 131, in training
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  File "/Library/Python/2.7/site-packages/tensorflow/python/training/optimizer.py", line 165, in minimize
    gate_gradients=gate_gradients)
  File "/Library/Python/2.7/site-packages/tensorflow/python/training/optimizer.py", line 205, in compute_gradients
    loss, var_list, gate_gradients=(gate_gradients == Optimizer.GATE_OP))
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/gradients.py", line 414, in gradients
    in_grads = _AsList(grad_fn(op_wrapper, *out_grads))
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/nn_grad.py", line 107, in _ReluGrad
    t = _VerifyTensor(op.inputs[0], op.name, "ReluGrad input is not finite.")
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/nn_grad.py", line 100, in _VerifyTensor
    verify_input = array_ops.check_numerics(t, message=msg)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/gen_array_ops.py", line 48, in check_numerics
    name=name)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/op_def_library.py", line 633, in apply_op
    op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 1710, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 988, in __init__
    self._traceback = _extract_stack()

...which was originally created as op u'conv1/Relu', defined at:
  File "featureProto.py", line 199, in <module>
    logits = inference(images_placeholder, keep_prob)
  File "featureProto.py", line 72, in inference
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/gen_nn_ops.py", line 506, in relu
    return _op_def_lib.apply_op("Relu", features=features, name=name)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/op_def_library.py", line 633, in apply_op
    op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 1710, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 988, in __init__
    self._traceback = _extract_stack()
```
      - エラー
- 第5回
  - データセット
    - 第4回同様
  - ニューラルネット
    - 学習率: 0.01
    - ステップ数: 100
    - 他は第4回同様
    - ログ:
      - step 96, training accuracy 1
step 97, training accuracy 0.998639
step 98, training accuracy 1
step 99, training accuracy 1
test accuracy 0.0795918
    - 正解率: 約8%
- 第6回
  - データセット
    - 第3回同様
  - ニューラルネット
    - 学習率: 1*e^-2(前回の約10倍)
    - 他は第3回同様
    - ログ: 
      - 

```
...
step 32, training accuracy 1
W tensorflow/core/common_runtime/executor.cc:1027] 0x7fde84c42720 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv1/add)]]
W tensorflow/core/common_runtime/executor.cc:1027] 0x7fde84c42720 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv2/Relu_grad/conv2/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv2/add)]]
W tensorflow/core/common_runtime/executor.cc:1027] 0x7fde84c42720 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/fc1/Relu_grad/fc1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](fc1/add)]]
Traceback (most recent call last):
  File "secondFeature.py", line 221, in <module>
    keep_prob: 0.5})
  File "/Library/Python/2.7/site-packages/tensorflow/python/client/session.py", line 345, in run
    results = self._do_run(target_list, unique_fetch_targets, feed_dict_string)
  File "/Library/Python/2.7/site-packages/tensorflow/python/client/session.py", line 419, in _do_run
    e.code)
tensorflow.python.framework.errors.InvalidArgumentError: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv1/add)]]
Caused by op u'gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics', defined at:
  File "secondFeature.py", line 198, in <module>
    train_op = training(loss_value, FLAGS.learning_rate)
  File "secondFeature.py", line 126, in training
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  File "/Library/Python/2.7/site-packages/tensorflow/python/training/optimizer.py", line 165, in minimize
    gate_gradients=gate_gradients)
  File "/Library/Python/2.7/site-packages/tensorflow/python/training/optimizer.py", line 205, in compute_gradients
    loss, var_list, gate_gradients=(gate_gradients == Optimizer.GATE_OP))
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/gradients.py", line 414, in gradients
    in_grads = _AsList(grad_fn(op_wrapper, *out_grads))
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/nn_grad.py", line 107, in _ReluGrad
    t = _VerifyTensor(op.inputs[0], op.name, "ReluGrad input is not finite.")
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/nn_grad.py", line 100, in _VerifyTensor
    verify_input = array_ops.check_numerics(t, message=msg)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/gen_array_ops.py", line 48, in check_numerics
    name=name)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/op_def_library.py", line 633, in apply_op
    op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 1710, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 988, in __init__
    self._traceback = _extract_stack()

...which was originally created as op u'conv1/Relu', defined at:
  File "secondFeature.py", line 194, in <module>
    logits = inference(images_placeholder, keep_prob)
  File "secondFeature.py", line 59, in inference
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/gen_nn_ops.py", line 506, in relu
    return _op_def_lib.apply_op("Relu", features=features, name=name)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/op_def_library.py", line 633, in apply_op
    op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 1710, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 988, in __init__
    self._traceback = _extract_stack()
```
    - エラー


- 現段階でのエリートモデル(第5回で生まれた分類器)を使って推薦部分を作ろうとおもう
  - とりあえずdivide.pyを使って, 分類器に投入した画像に近い画像を5枚出力させ, 目視確認してみた
  - 一番はっきり顔を見せてくれた5ちゃんを使って
    - ![5ちゃんのある1枚は順番に](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/5の4.jpeg)
      - ![この子に似ている](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/172の1.jpeg)
      - ![この子に似ている](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/216の2.jpeg)
      - ![この子に似ている](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/181の3.jpeg)
      - ![この子に似ている](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/200の4.jpeg)
      - ![この子に似ている](https://github.com/egroup-develop/TensorFlow/tree/feature-tensorboard/imgs/143の3.jpeg)
  - 所感
    - ほーわからなくもない
    - しかし一人当たり8枚しか教師として与えていないのにこの結果はすごいのでは...?
    - これ絶対おもしろいサービスになる
    - 一つのクラスに対してデータの数さえ十分あれば精度はよくなる. パラメータは現段階モデルが最高だと思う. あとは数. 動画ですね!!!



