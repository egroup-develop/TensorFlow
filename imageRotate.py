# -*- coding: utf-8 -*-
 
import cv2
import numpy as np
import sys
from PIL import Image
import os

if __name__ == '__main__': 
  # 正規化したい画像のクラスが格納されている2階層ディレクトリを引数として与える
  os.chdir(sys.argv[1])

  def imgRot(img_path):
    img_src = cv2.imread(img_path, 1)

    center = tuple(np.array([img_src.shape[1] * 0.5, img_src.shape[0] * 0.5]))
    size = tuple(np.array([img_src.shape[1], img_src.shape[0]]))
    
    # 10度ずつ回転させる
    angle = 10.0
 
    # スケール1倍
    scale = 1.0

    while angle <= 360:
      rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
      img_rot = cv2.warpAffine(img_src, rotation_matrix, size, flags=cv2.INTER_CUBIC)

      img_path = img_path.replace(".jpeg", "")
      cv2.imwrite(img_path + "_" + str(int(angle)) + "rotated.jpeg", img_rot)

      angle += 10.0

  for i in range(0, 328):
    dirName = str(i)
    files = os.listdir(dirName)

    for file in files:
      j = 1

      while j < 5:
        if file == "image_" + str(j) + "_convert.jpeg" or file == "image_" + str(j) + "_origin_convert.jpeg":
          imgRot(dirName + "/" + file)

        j += 1
