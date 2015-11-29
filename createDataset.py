# -*- coding: utf-8 -*-
import os
import sys

trainTitle = "train.txt"
testTitle = "test.txt"

#データセットテキストファイルに変換したい2階層ディレクトリを引数に与える
os.chdir(sys.argv[1])

for i in range(0, 100):
  trainText = ""
  testText = ""
  dirName = str(i)

  files = os.listdir(dirName)

  j = 0
  for file in files:
    if j < len(files) - 1:
      trainText += sys.argv[1] + dirName + "/" + file + " " + dirName + "\n"
    else:
      testText += sys.argv[1] + dirName + "/" + file + " " + dirName + "\n"
    j += 1
  
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
