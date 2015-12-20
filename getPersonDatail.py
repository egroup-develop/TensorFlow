# -*- coding: utf-8 -*-
import urllib
import urllib2
import json

u = 'start=4&limit=400&category=Daily%2BLoGiRL'
title = "personDataList.json"
personList = {}
jsonList = {}
urlList = []

try:
  r = urllib2.urlopen(u)
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
 
    nameList.append(name)
    print "ID: " + id
    print u"名前: " + nameList[len(nameList)-1]
    personList["Name"] = nameList[len(nameList)-1]

    index = titles[len(titles) - 3:]
    if index[0] == u"0":
      if index[1] == u"0":
        index = index[2]    
      else:
        index = index[1] + index[2]

    print "index: " + index

    k = 1
    for url in urls:
      if k < 5:
        print "URL: " + url
        urlList.append(url)

      k += 1

    personList["Url"] = urlList
    jsonList[str(int(index) - 1)] = personList

    personList = {}
    urlList = []

  with open(title, "w") as f:
    json.dump(jsonList, f, sort_keys = True, indent = 4)

finally: 
  r.close()
