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
        for f in os.listdir():
            xmldoc = minidom.parse(f)
            itemlist = xmldoc.getElementsByTagName('word')

            wordList = []
            for s in itemlist:
                for key in s.attributes.keys():
                    dct = {}
                    dct[key] = s.attributes[key].value
                    wordList.append(dct)
                
            for w in wordList:
                for key in w.keys():
                    print(w[key])
                print()