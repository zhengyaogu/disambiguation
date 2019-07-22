from xml.dom import minidom
from cd import Cd
import os

"""
The Cd context manager assumes that we will only be navigating
to more specific directories, so it needs to start outside of each 
of the folders. 
"""
os.chdir('..')
with Cd('word_sense_disambigation_corpora'):
    with Cd('masc/written/blog/'):
        files = os.listdir()
        xmldoc = minidom.parse(files[0])
        itemlist = xmldoc.getElementsByTagName('word')
        print(len(itemlist))
        print(itemlist[0].attributes['text'].value)

        wordList = []
        for s in itemlist:
            dct = {}
            dct["text"] = s.attributes['text'].value
            wordList.append(dct)
        
        for w in wordList:
            print(dct["text"])