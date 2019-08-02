from multiprocessing import Pool, Queue, cpu_count, Manager
from compare import createSentenceDictionaries
import time
import sys
import json


def writerToQueue(document_queue_list):
    document, formatted_data_queue = document_queue_list[0], document_queue_list[1]
    doc_dict = {}
    doc_dict["docname"] = document["docname"]
    doc_dict["doc"] = []
    createSentenceDictionaries(document["doc"], doc_dict["doc"])
    formatted_data_queue.put(doc_dict)
    print("finished processing the document: " + document["docname"])

if __name__ == '__main__':
   
    with open("googledata.json") as json_file:
        data = json.load(json_file)

    p = Pool(processes=cpu_count()*2)
    m = Manager()
    q = m.Queue()
    bundle = []
    for document in data:
        bundle.append([document, q])
    p.map(writerToQueue, bundle)
    

    formatted_data = []
    while not q.empty():
        formatted_data.append(q.get())
    with open("completedataV2.json", "w") as json_file:
        json.dump(formatted_data, json_file, indent=4)
    
