# -*- coding: utf-8 -*-
 
import cv2
import numpy as np
import sys
from PIL import Image

if __name__ == '__main__': 
  img_path = sys.argv[1]
#  angle = 0

#  img_src = Image.open(img_path)
#  img_path = img_path.replace(".jpeg", "")

#  i = 10
#  while i <= 360:
#    angle = i
#    tmp = img_src.rotate(angle)
#    tmp.save(img_path + "_" + str(angle) + "rotated.jpeg")
#    i += 10


  #以下opencvでの画像回転
  # 画像読み込み
  img_src = cv2.imread(img_path, 1)

  # 画像の中心位置
  # 今回は画像サイズの中心を中心位置としている
  #center = tuple(np.array([img_src.shape[1] * 0.5, img_src.shape[0] + 0.5]))
  center = tuple(np.array([img_src.shape[1] * 0.5, img_src.shape[0] * 0.5]))

  # 画像サイズの取得(横, 縦)
  size = tuple(np.array([img_src.shape[1], img_src.shape[0]]))

  # 回転させたい角度
  # ラジアンではなく角度(°)
  angle = 10.0

  # 拡大比率
  scale = 1.0

  while angle <= 360:
    # 以上の条件から2次元の回転行列を計算. 回転変換行列の算出
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # アフィン変換
    img_rot = cv2.warpAffine(img_src, rotation_matrix, size, flags=cv2.INTER_CUBIC)

    img_path = img_path.replace(".jpeg", "")
    # 保存
    cv2.imwrite(img_path + "_" + str(int(angle)) + "rotated.jpeg", img_rot)

    angle += 10.0
