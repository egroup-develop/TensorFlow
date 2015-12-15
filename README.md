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

- 第7回
  - デーロル328人, 1人につき288枚
    - 内訳
      - 1人4枚の画像があり, 各1枚を正規化(コントラストの強調)して数を2枚に増やし, それを360度まで10度ずつ回転させたので1枚を72枚まで増やしたx4
  - データセット
    - 訓練: 70848
    - 評価: 23616
  - ニューラルネット
    - 学習率 0.01
    - 畳み込み 2回
    - pooling 2回
    - ステップ数 2000
    - バッチサイズ 100
  - 結果
    - 音無くPCがシャットダウンした...(メモリ4GBのMBA)
    - これまでのエラーと同じなんだが, AdamOptimizerでのエラーみたい. 学習率を低くしろだとか初期重みを変えろだとかいろいろ言われている. とにかく言えることはプログラムで解決する問題ではないみたい..?
      - [Tensorflow crashed when using AdamOptimizer](https://github.com/tensorflow/tensorflow/issues/323)
- 第8回
  - 再びデーロル245人で. 1人につき, 正規化した画像を45度-135度まで9度ずつずらした画像11枚
    - 訓練 8085
    - 評価 2695
  - ニューラルネット
    - 第5回同様
  - 結果
    - step 0, training accuracy 0.00693584
step 1, training accuracy 0.0148625
step 2, training accuracy 0.0132524
step 3, training accuracy 0.0158534
step 4, training accuracy 0.02378
step 5, training accuracy 0.0282388
step 6, training accuracy 0.0226653
step 7, training accuracy 0.019569
step 8, training accuracy 0.0261333
step 9, training accuracy 0.0294773
step 10, training accuracy 0.0313351
step 11, training accuracy 0.0416151
step 12, training accuracy 0.0497894
step 13, training accuracy 0.0465692
step 14, training accuracy 0.0574684
step 15, training accuracy 0.0666336
step 16, training accuracy 0.0718355
step 17, training accuracy 0.0755512
step 18, training accuracy 0.0848402
W tensorflow/core/common_runtime/executor.cc:1027] 0x7fdf49c00b90 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv1/add)]]
W tensorflow/core/common_runtime/executor.cc:1027] 0x7fdf49c00b90 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv2/Relu_grad/conv2/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv2/add)]]
W tensorflow/core/common_runtime/executor.cc:1027] 0x7fdf49c00b90 Compute status: Invalid argument: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/fc1/Relu_grad/fc1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](fc1/add)]]
Traceback (most recent call last):
  File "featureProto.py", line 245, in <module>
    keep_prob: 0.5})
  File "/Library/Python/2.7/site-packages/tensorflow/python/client/session.py", line 345, in run
    results = self._do_run(target_list, unique_fetch_targets, feed_dict_string)
  File "/Library/Python/2.7/site-packages/tensorflow/python/client/session.py", line 419, in _do_run
    e.code)
tensorflow.python.framework.errors.InvalidArgumentError: ReluGrad input is not finite. : Tensor had NaN values
	 [[Node: gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics = CheckNumerics[T=DT_FLOAT, message="ReluGrad input is not finite.", _device="/job:localhost/replica:0/task:0/cpu:0"](conv1/add)]]
Caused by op u'gradients/conv1/Relu_grad/conv1/Relu/CheckNumerics', defined at:
  File "featureProto.py", line 222, in <module>
    train_op = training(loss_value, FLAGS.learning_rate)
  File "featureProto.py", line 138, in training
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
  File "featureProto.py", line 218, in <module>
    logits = inference(images_placeholder, keep_prob)
  File "featureProto.py", line 77, in inference
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/gen_nn_ops.py", line 506, in relu
    return _op_def_lib.apply_op("Relu", features=features, name=name)
  File "/Library/Python/2.7/site-packages/tensorflow/python/ops/op_def_library.py", line 633, in apply_op
    op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 1710, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/Library/Python/2.7/site-packages/tensorflow/python/framework/ops.py", line 988, in __init__
    self._traceback = _extract_stack()
      - 原因
        - ReLU関数が無限大に達した時エラーになる. つまりReLUに与える重みが無限に発散した時(もっと正確に言うと, loss関数の結果がnanになった時). この重み, 正常な状態では[-1, 1]の値を保つ. つまり, エラー時にはこの部分で 0/0 や 0x無限 のような不正な浮動小数演算を行っている. 
      - 対策
        - 適当な(活性化関数に与える重みが発散しないような)超パラメータを探すこと. 学習率や重み等を試行錯誤する必要がある. ⇒ DNNの醍醐味
          - 学習率についてはAdamの場合, 1e^-4が安定して学習できた(以下の実験にて実証). これより低くても高くてもダメだった.

- 第9回
  - デーロル245人で. 1人につき, 正規化した画像を45度-135度まで9度ずつずらした画像11枚
    - 訓練 8085
    - 評価 2695
  - ニューラルネット
    - 第8回同様
    - 重みのみ変更あり
      - 畳み込み5*5を3*3に変更
  - 結果
    - step 0, training accuracy 0.00655535
step 1, training accuracy 0.00618429
step 2, training accuracy 0.0111317
step 3, training accuracy 0.0174397
step 4, training accuracy 0.0213977
step 5, training accuracy 0.0298083
...
step 78, training accuracy 0.959926
step 79, training accuracy 0.959926
step 80, training accuracy 0.962028
step 81, training accuracy 0.966234
82ステップ目でエラー

- 第10回
  - デーロル245人で. 1人につき, 正規化した画像を45度-135度まで9度ずつずらした画像11枚
    - 訓練 8085
    - 評価 2695
  - ニューラルネット
    - 第9回同様
    - 学習率のみ変更あり
      - 0.01を1e^-8にした
  - 結果
    - step 0, training accuracy 0.00581323
step 1, training accuracy 0.00593692
step 2, training accuracy 0.00593692
step 3, training accuracy 0.00593692
step 4, training accuracy 0.00593692
step 5, training accuracy 0.00593692
step 6, training accuracy 0.00593692
step 7, training accuracy 0.00593692
step 8, training accuracy 0.00593692
step 9, training accuracy 0.00593692
step 10, training accuracy 0.00593692
step 11, training accuracy 0.00593692
step 12, training accuracy 0.00593692
step 13, training accuracy 0.00593692
step 14, training accuracy 0.00593692
step 15, training accuracy 0.00593692
step 16, training accuracy 0.00593692
step 17, training accuracy 0.00593692
step 18, training accuracy 0.00593692
step 19, training accuracy 0.00593692
step 20, training accuracy 0.00593692
step 21, training accuracy 0.00593692
step 22, training accuracy 0.00593692
step 23, training accuracy 0.00593692
step 24, training accuracy 0.00606061
step 25, training accuracy 0.00606061
step 26, training accuracy 0.00606061
step 27, training accuracy 0.00606061
step 28, training accuracy 0.00606061
step 29, training accuracy 0.00606061
step 30, training accuracy 0.00606061
step 31, training accuracy 0.00606061
step 32, training accuracy 0.00606061
step 33, training accuracy 0.00606061
step 34, training accuracy 0.00606061
step 35, training accuracy 0.00618429
step 36, training accuracy 0.00618429
step 37, training accuracy 0.00618429
step 38, training accuracy 0.00618429
step 39, training accuracy 0.00618429
step 40, training accuracy 0.00618429
step 41, training accuracy 0.00618429
step 42, training accuracy 0.00618429
step 43, training accuracy 0.00618429
step 44, training accuracy 0.00618429
step 45, training accuracy 0.00618429
step 46, training accuracy 0.00618429
step 47, training accuracy 0.00618429
step 48, training accuracy 0.00618429
step 49, training accuracy 0.00618429
step 50, training accuracy 0.00618429
step 51, training accuracy 0.00618429
step 52, training accuracy 0.00618429
step 53, training accuracy 0.00618429
step 54, training accuracy 0.00618429
step 55, training accuracy 0.00618429
step 56, training accuracy 0.00618429
step 57, training accuracy 0.00618429
step 58, training accuracy 0.00618429
step 59, training accuracy 0.00618429
step 60, training accuracy 0.00618429
step 61, training accuracy 0.00618429
step 62, training accuracy 0.00618429
step 63, training accuracy 0.00618429
step 64, training accuracy 0.00630798
step 65, training accuracy 0.00630798
step 66, training accuracy 0.00630798
step 67, training accuracy 0.00630798
step 68, training accuracy 0.00630798
step 69, training accuracy 0.00630798
step 70, training accuracy 0.00630798
step 71, training accuracy 0.00630798
step 72, training accuracy 0.00630798
step 73, training accuracy 0.00630798
step 74, training accuracy 0.00630798
step 75, training accuracy 0.00630798
step 76, training accuracy 0.00630798
step 77, training accuracy 0.00630798
step 78, training accuracy 0.00630798
step 79, training accuracy 0.00630798
step 80, training accuracy 0.00630798
step 81, training accuracy 0.00630798
step 82, training accuracy 0.00630798
step 83, training accuracy 0.00630798
step 84, training accuracy 0.00630798
step 85, training accuracy 0.00630798
step 86, training accuracy 0.00630798
step 87, training accuracy 0.00630798
step 88, training accuracy 0.00630798
step 89, training accuracy 0.00630798
step 90, training accuracy 0.00630798
step 91, training accuracy 0.00630798
step 92, training accuracy 0.00630798
step 93, training accuracy 0.00630798
step 94, training accuracy 0.00630798
step 95, training accuracy 0.00630798
step 96, training accuracy 0.00630798
step 97, training accuracy 0.00630798
step 98, training accuracy 0.00630798
step 99, training accuracy 0.00630798
step 100, training accuracy 0.00630798
=step 101, training accuracy 0.00630798
step 102, training accuracy 0.00618429
step 103, training accuracy 0.00618429
step 104, training accuracy 0.00618429
step 105, training accuracy 0.00618429
step 106, training accuracy 0.00618429
step 107, training accuracy 0.00618429
step 108, training accuracy 0.00618429
step 109, training accuracy 0.00618429
step 110, training accuracy 0.00618429
step 111, training accuracy 0.00618429
step 112, training accuracy 0.00618429
step 113, training accuracy 0.00618429
step 114, training accuracy 0.00618429
step 115, training accuracy 0.00618429
step 116, training accuracy 0.00618429
      - 4時間の学習経過. ステップ2000にしたので終わりがみえない. 学習率が低すぎて計算時間が膨大に. 打ち切り.

- 第11回
  - デーロル245人で. 1人につき, 正規化した画像を45度-135度まで9度ずつずらした画像11枚
    - 訓練 8085
    - 評価 2695
  - ニューラルネット
    - 第10回同様
    - 学習率のみ変更あり
      - 1e^-5にした
  - 結果
    - step 0, training accuracy 0.00420532
step 1, training accuracy 0.00482375
step 2, training accuracy 0.00519481
step 3, training accuracy 0.00507112
step 4, training accuracy 0.00544218
step 5, training accuracy 0.00494743
step 6, training accuracy 0.00457638
step 7, training accuracy 0.00494743
step 8, training accuracy 0.00556586
step 9, training accuracy 0.00568955
step 10, training accuracy 0.00556586
step 11, training accuracy 0.00606061
step 12, training accuracy 0.00643166
step 13, training accuracy 0.00667904
step 14, training accuracy 0.00729746
step 15, training accuracy 0.00779221
step 16, training accuracy 0.00766852
step 17, training accuracy 0.00754484
step 18, training accuracy 0.00803958
step 19, training accuracy 0.00841064
step 20, training accuracy 0.00865801
step 21, training accuracy 0.00952381
step 22, training accuracy 0.00977118
step 23, training accuracy 0.0118738
step 24, training accuracy 0.0132344
step 25, training accuracy 0.0145949
step 26, training accuracy 0.0155844
step 27, training accuracy 0.0170686
step 28, training accuracy 0.0175634
step 29, training accuracy 0.0194187
step 30, training accuracy 0.0201608
step 31, training accuracy 0.021274
step 32, training accuracy 0.0223871
step 33, training accuracy 0.0238714
step 34, training accuracy 0.0253556
step 35, training accuracy 0.0251082
step 36, training accuracy 0.0264688
step 37, training accuracy 0.0278293
step 38, training accuracy 0.0305504
step 39, training accuracy 0.0317873
step 40, training accuracy 0.0337662
step 41, training accuracy 0.034261
step 42, training accuracy 0.0368584
step 43, training accuracy 0.0380952
step 44, training accuracy 0.040569
step 45, training accuracy 0.04094
step 46, training accuracy 0.0423006
step 47, training accuracy 0.0431664
step 48, training accuracy 0.04329
step 49, training accuracy 0.0447743
step 50, training accuracy 0.0481138
step 51, training accuracy 0.0497217
step 52, training accuracy 0.0493506
step 53, training accuracy 0.0512059
step 54, training accuracy 0.0528139
step 55, training accuracy 0.0547928
step 56, training accuracy 0.0565244
step 57, training accuracy 0.057885
step 58, training accuracy 0.0587508
step 59, training accuracy 0.0599876
step 60, training accuracy 0.0620903
step 61, training accuracy 0.0643166
step 62, training accuracy 0.0654298
step 63, training accuracy 0.0674088
step 64, training accuracy 0.0683983
step 65, training accuracy 0.0695114
step 66, training accuracy 0.0703772
step 67, training accuracy 0.0719852
step 68, training accuracy 0.0740878
step 69, training accuracy 0.0761905
step 70, training accuracy 0.0768089
step 71, training accuracy 0.0781695
step 72, training accuracy 0.0789116
step 73, training accuracy 0.0803958
step 74, training accuracy 0.0817563
step 75, training accuracy 0.0831169
step 76, training accuracy 0.0852195
step 77, training accuracy 0.0879406
step 78, training accuracy 0.0884354
step 79, training accuracy 0.0915275
step 80, training accuracy 0.094867
step 81, training accuracy 0.0951144
step 82, training accuracy 0.0957328
step 83, training accuracy 0.100186
step 84, training accuracy 0.101175
step 85, training accuracy 0.101917
step 86, training accuracy 0.104638
step 87, training accuracy 0.107359
step 88, training accuracy 0.108349
step 89, training accuracy 0.112925
step 90, training accuracy 0.113791
step 91, training accuracy 0.11713
step 92, training accuracy 0.117625
step 93, training accuracy 0.119481
step 94, training accuracy 0.121583
step 95, training accuracy 0.122944
step 96, training accuracy 0.126531
step 97, training accuracy 0.128262
step 98, training accuracy 0.129623
step 99, training accuracy 0.132344
step 100, training accuracy 0.134447
step 101, training accuracy 0.136178
step 102, training accuracy 0.138281
step 103, training accuracy 0.142239
step 104, training accuracy 0.145083
      - 5時間経過の結果. 学習による正解率はとてもゆるやかに上昇. それにしても時間がかかる. ターミナルが予期せぬ終了して学習がオワタ

- 第12回
  - デーロル245人で. 1人につき, 正規化した画像を45度-135度まで9度ずつずらした画像11枚
    - 訓練 8085
    - 評価 2695
  - ニューラルネット
    - 第11回同様
    - 学習率のみ変更あり
      - 1e^-3にした
  - 結果
    - step 0, training accuracy 0.00482375
step 1, training accuracy 0.00494743
step 2, training accuracy 0.00420532
step 3, training accuracy 0.00408163
step 4, training accuracy 0.00383426
step 5, training accuracy 0.00408163
step 6, training accuracy 0.00408163
step 7, training accuracy 0.00507112
step 8, training accuracy 0.00395795
step 9, training accuracy 0.00408163
step 10, training accuracy 0.00408163
step 11, training accuracy 0.00482375
step 12, training accuracy 0.00408163
step 13, training accuracy 0.00482375
step 14, training accuracy 0.00383426
step 15, training accuracy 0.00395795
step 16, training accuracy 0.00333952
step 17, training accuracy 0.00470006
step 18, training accuracy 0.00408163
step 19, training accuracy 0.004329
step 20, training accuracy 0.00494743
step 21, training accuracy 0.00581323
step 22, training accuracy 0.00371058
step 23, training accuracy 0.00408163
step 24, training accuracy 0.00531849
step 25, training accuracy 0.00618429
step 26, training accuracy 0.004329
step 27, training accuracy 0.00593692
step 28, training accuracy 0.00507112
step 29, training accuracy 0.00445269
step 30, training accuracy 0.00371058
step 31, training accuracy 0.00457638
step 32, training accuracy 0.00309215
step 33, training accuracy 0.00482375
step 34, training accuracy 0.00408163
step 35, training accuracy 0.00420532
step 36, training accuracy 0.004329
step 37, training accuracy 0.00420532
step 38, training accuracy 0.004329
step 39, training accuracy 0.00445269
step 40, training accuracy 0.00420532
step 41, training accuracy 0.00482375
step 42, training accuracy 0.00395795
step 43, training accuracy 0.00395795
step 44, training accuracy 0.00395795
step 45, training accuracy 0.00408163
step 46, training accuracy 0.00408163
step 47, training accuracy 0.00408163
step 48, training accuracy 0.004329
step 49, training accuracy 0.00568955
step 50, training accuracy 0.00507112
step 51, training accuracy 0.00655535
step 52, training accuracy 0.00606061
step 53, training accuracy 0.00902907
step 54, training accuracy 0.00742115
step 55, training accuracy 0.00816326
step 56, training accuracy 0.00878169
step 57, training accuracy 0.0107607
step 58, training accuracy 0.0096475
step 59, training accuracy 0.00940012
step 60, training accuracy 0.00408163
step 61, training accuracy 0.00890538
step 62, training accuracy 0.00371058
step 63, training accuracy 0.00383426
step 64, training accuracy 0.00680272
      - 第11回と時同じくしてターミナルがクラッシュしてオワタ. 5時間経過の結果. 正解率がまったく上昇していない. あきらかに過学習している. これまでの実験を通して, 本データセット, NN構想には学習率1e^-4が最も適していると言える. したがって今後は重みでチューニングしていくことにする(loss値が非数にならぬように)
