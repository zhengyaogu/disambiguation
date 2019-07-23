from xml.dom import minidom
from cd import Cd
import os
import json
import pprint

"""
Returns a list of dictionaries, where each dictionary contains all the key 
value pairs of a word element.
"""
def parseXmlFile(f):
    print(f)
    xmldoc = minidom.parse(f)
    header = xmldoc.getElementsByTagName('SimpleWsdDoc')[0]
    docname = header.attributes['name'].value

    itemlist = xmldoc.getElementsByTagName('word')

    document_dictionary = {}
    document_dictionary["docname"] = docname
    wordList = []
    for word in itemlist:
        dct = {}
        for key in word.attributes.keys():
            
            dct[key] = word.attributes[key].value
        wordList.append(dct)
    
    document_dictionary["doc"]= wordList

    return document_dictionary

"""
Returns a list of dictionaries, containing all the word data in all the xml files
in the specified directory and all of its sub directories.
"""
def parseDirectory(directory):
    data = []
    for root, dirs, files in os.walk(directory_to_walk):
        with Cd(root):
            for f in files:
                if f.endswith(".xml"):
                    data.append(parseXmlFile(f))
    return data

if __name__ == "__main__":
    os.chdir('..')
    directory_to_walk = os.path.join(os.getcwd(), 'word_sense_disambigation_corpora')
    data = parseDirectory(directory_to_walk)
    filename = "googledata.json"
    os.chdir(os.path.join(os.getcwd(), "disambiguation" ))
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

