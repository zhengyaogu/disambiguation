from xml.dom import minidom
from cd import Cd
import os

"""
Returns a list of dictionaries, where each dictionary contains all the key 
value pairs of a word element.
"""
def parseXmlFile(f):
    print(f)
    xmldoc = minidom.parse(f)
    itemlist = xmldoc.getElementsByTagName('word')

    wordList = []
    for word in itemlist:
        for key in word.attributes.keys():
            dct = {}
            dct[key] = word.attributes[key].value
            wordList.append(dct)
    return wordList

"""
Returns a list of dictionaries, containing all the word data in all the xml files
in the specified directory and all of its sub directories.
"""
def parseDirectory(directory):
    data = []
    for root, dirs, files in os.walk(directory_to_walk):
        with Cd(root):
            print(os.getcwd())
            for f in files:
                if f.endswith(".xml"):
                    data.extend(parseXmlFile(f))
    return data



if __name__ == "__main__":
    os.chdir('..')
    directory_to_walk = os.path.join(os.getcwd(), 'word_sense_disambigation_corpora')
    data = parseDirectory(directory_to_walk)


    for word in data:
        line = ""
        for key in word.keys():
            line += key + ": " + word[key] + " "
        print(line) 
                
        
        
