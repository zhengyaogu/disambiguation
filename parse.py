from xml.dom import minidom

xmldoc = minidom.parse('word_sense_disambigation_corpora/masc/written/blog/Acephalous-Cant-believe.xml')
itemlist = xmldoc.getElementsByTagName('word')
print(len(itemlist))
print(itemlist[0].attributes['text'].value)
for s in itemlist:
    print(s.attributes['text'].value)