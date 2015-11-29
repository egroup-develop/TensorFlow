# -*- coding: utf-8 -*-
import urllib
import urllib2
import json
import os
import sys

u = ''
title = "LogirlImages"

try:
  r = urllib2.urlopen(u)
  
  try: 
    os.mkdir(title)
    os.chdir(title)
  except: 
    sys.exit()

  i = 0
  for ary in json.loads(r.read()):
    id = ary['Id']
    urls = ary['Url']
    num = str(i)

    try:
      os.mkdir(num)
      os.chdir(num)
    except: 
      sys.exit()

    print "ID: " + id

    j = 1
    for url in urls:
      print "URL: " + url

      filename = "image_" + str(j) + ".png"
      savePath = os.path.join(os.getcwd(), os.path.basename(filename))
      urllib.urlretrieve(url, savePath)
      j += 1

    os.chdir("../")
    i += 1


finally: 
  r.close()
