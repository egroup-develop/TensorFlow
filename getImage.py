# -*- coding: utf-8 -*-
import urllib
import urllib2
import json
import os
import sys

u = ''h
title = "DailyLogirlImages"
nameStr = u'#!/usr/bin/env python' + u'\n' + u'# -*- coding: utf-8 -*-' + u'\n' + u'import sys' + u'\n' + u'if __name__ == "__main__":' + u'\n' + u'  a = {'
log = ""

try:
  r = urllib2.urlopen(u)
  
  try: 
    os.mkdir(title)
    os.chdir(title)
  except: 
    sys.exit()

  name = ""
  nameList = []
  nameStart = 0
  nameEnd = 0
  index = ""

  for ary in json.loads(r.read()):
    id = ary['Id']
    titles = ary['Title']
    urls = ary['Url']

    if titles[0] == u"【":
      eval = True
      i = 0

      while eval:
        if titles[i] == u"】":
          eval = False
        i += 1

      nameStart = i

    eval = True
    parenthesisEval = False

    for i in range(len(titles)):
      if titles[i] == u"（":
        parenthesisEval = True
        nameEnd = i
        break

    i = 0
    if not parenthesisEval:
      while eval:
        if titles[i] == u"D":
          nameEnd = i - 1
          eval = False
        i += 1

    for i in range(nameStart, nameEnd):
      name += titles[i]

    print u"名前: " + name

    index = titles[len(titles) - 3:]
    if index[0] == u"0":
      if index[1] == u"0":
        index = index[2]    
      else:
        index = index[1] + index[2]

    nameList.append(str(int(index) - 1)) 
    nameList.append(name)

    print "index: " + index

    try:
      os.mkdir(str(int(index) - 1))
      os.chdir(str(int(index) - 1))
    except: 
      sys.exit()

    print "ID: " + id

    j = 1
    k = 1
    for url in urls:
      if k < 5:
        print "URL: " + url

        filename = "image_" + str(j)
        savePath = os.path.join(os.getcwd(), os.path.basename(filename))
        urllib.urlretrieve(url, savePath)
      j += 1
      k += 1
    if j != 4:
      log += name

    os.chdir("../")
    i += 1
    
    name = ""
    nameStart = 0

  for i in range(len(nameList)):
    if i == len(nameList) - 1:
      nameStr += u'"' + nameList[i] + u'"}' + u'\n' + u'  print str(a[int(sys.argv[1])])' + u'\n'
    elif i % 2 == 0:
      nameStr += u'"' + nameList[i]  + u'"' + u':'
    else:
      nameStr += nameList[i] + u'"' + u', '
 

  try:
    os.chdir("../")
    f = open("getName.py", "a+")
    f.write(nameStr.encode('utf-8'))
    f.close()
  except:
    sys.exit()
    
  try:
    f = open("fall_log.txt", "a+")
    f.write(log)
    f.close()
  except:
    sys.exit()

finally: 
  r.close()
