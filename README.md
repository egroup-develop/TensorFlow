# 実験結果
- 第1回
 - データセット: デ**ーロ**ルの100人. 1人につき4枚
  - 訓練: 300枚
  - 評価: 100枚
 - 正規化: していない(顔切り抜きスクリプトは作った. 最終確認は目視で, 単にそれが大変なので後回しにした)
 - ニューラルネット
  - 畳み込み 2層
  - プーリング 2層
  - 活性化関数fc 2層
   - fc1層はドロップアウトさせている
   - fc2層の値を取り出してソフトマックス関数にて正規化->これが特徴量
  - 入力は28x28x3
  - ステップ数は200
 - 学習
  - 要した時間: 約15分
  - ログ: 
pre(
step 0, training accuracy 0.02
step 1, training accuracy 0.0166667
step 2, training accuracy 0.0266667
step 3, training accuracy 0.03
step 4, training accuracy 0.0566667
step 5, training accuracy 0.11
step 6, training accuracy 0.136667
step 7, training accuracy 0.186667
step 8, training accuracy 0.253333
step 9, training accuracy 0.353333
step 10, training accuracy 0.44
step 11, training accuracy 0.463333
step 12, training accuracy 0.573333
step 13, training accuracy 0.643333
step 14, training accuracy 0.69
step 15, training accuracy 0.75
step 16, training accuracy 0.796667
step 17, training accuracy 0.873333
step 18, training accuracy 0.883333
step 19, training accuracy 0.916667
step 20, training accuracy 0.926667
step 21, training accuracy 0.94
step 22, training accuracy 0.96
step 23, training accuracy 0.963333
step 24, training accuracy 0.983333
step 25, training accuracy 0.986667
step 26, training accuracy 0.983333
step 27, training accuracy 0.993333
step 28, training accuracy 0.996667
step 29, training accuracy 0.996667
step 30, training accuracy 0.996667
step 31, training accuracy 0.996667
step 32, training accuracy 0.996667
step 33, training accuracy 0.996667
step 34, training accuracy 0.996667
step 35, training accuracy 0.996667
step 36, training accuracy 1
step 37, training accuracy 1
step 38, training accuracy 1
step 39, training accuracy 1
step 40, training accuracy 1
...
step 192, training accuracy 1
step 193, training accuracy 1
step 194, training accuracy 1
step 195, training accuracy 1
step 196, training accuracy 1
step 197, training accuracy 1
step 198, training accuracy 1
step 199, training accuracy 1
test accuracy 0.02
)
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
