# -*- coding: utf-8 -*-
import os
import sys

trainTitle = "train.txt"
testTitle = "test.txt"

#データセットテキストファイルに変換したい2階層ディレクトリを引数に与える. 末尾スラッシュありで
os.chdir(sys.argv[1])

for i in range(0, 328):
  trainText = ""
  testText = ""
  dirName = str(i)

  files = os.listdir(dirName)

#  j = 0
  eval = 0
  for file in files:
    if file.find("image_4_") > -1:
      if file.find("_convert_") > -1:
        if file.find("rotated.jpeg") > -1:
          testText += sys.argv[1] + dirName + "/" + file + " " + dirName + "\n"
          eval = 1
#    elif file.endswith("convert.jpeg", 15, len(file)) or file.endswith("convert.jpeg", 8, len(file)):
    elif file.endswith("rotated.jpeg"):
      if eval == 0:
        trainText += sys.argv[1] + dirName + "/" + file + " " + dirName + "\n"
    eval = 0

#    if j < len(files) - 1:
#      trainText += sys.argv[1] + dirName + "/" + file + " " + dirName + "\n"
#    else:
#      testText += sys.argv[1] + dirName + "/" + file + " " + dirName + "\n"
#    j += 1
  
  
  try:   
    f = open(trainTitle, "a+")
    f.write(trainText)
    f.close()
  except:
    sys.exit()

  try:   
    f = open(testTitle, "a+")
    f.write(testText)
    f.close()
  except:
    sys.exit()
