# -*- coding:utf-8 -*-
import numpy
import cv2
import sys
import os

cascade_path = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
image_path = sys.argv[1]

color = (255, 255, 255) #白

#ファイル読み込み
image = cv2.imread(image_path)

#グレースケール変換なんだけど, has no attribute 'cv' エラーが出るのでコメントアウトする. 画像認識の高速化のためのグレースケーリングなんでまあいい
#image_gray = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)

#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

#物体認識（顔認識）の実行
facerect = cascade.detectMultiScale(
                image, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
if len(facerect) <= 0:
    print os.getcwd() + "/" + image_path
    exit()

#検出した顔を囲む矩形の作成
#for rect in facerect:
#    cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)

#認識結果の保存
#image_path = image_path.replace(".png", "")
image_path = image_path.replace(".jpeg", "")

#確認用
#cv2.imwrite(image_path + "_detected.png", image)
for rect in facerect:
    #cv2.imwrite(image_path + ".png", image[rect])
    print rect
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    
    # img[y: y + h, x: x + w] 
    #cv2.imwrite(image_path + ".png", image[y:y+h, x:x+w])
    cv2.imwrite(image_path + ".jpeg", image[y:y+h, x:x+w])
