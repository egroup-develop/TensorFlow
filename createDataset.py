# -*- coding: utf-8 -*-
import os
import sys

title = "train.txt"

#データセットテキストファイルに変換したい2階層ディレクトリを引数に与える
os.chdir(sys.argv[1])

for i in range(0, 100):
  textCode = ""
  dirName = str(i)

  files = os.listdir(dirName)

  for file in files:
    textCode += sys.argv[1] + dirName + "/" + file + " " + dirName + "\n"
  
  try:   
    f = open(title, "a+")
    f.write(textCode)
    f.close()
  except:
    sys.exit()
