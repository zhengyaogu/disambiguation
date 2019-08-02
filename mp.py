from multiprocessing import Process, Queue
from compare import createSentenceDictionaries
import time
import sys
import json


def writerToQueue(tuplee):
    documents, formatted_data_queue = tuplee
    for document in documents:
        doc_dict = {}
        doc_dict["docname"] = document["docname"]
        doc_dict["doc"] = []
        createSentenceDictionaries(document["doc"], doc_dict["doc"])
        formatted_data_queue.put(doc_dict)
        print("finished processing the document: " + document["docname"])



if __name__ == '__main__':
    data_queue = Queue()
    with open("googledata.json") as json_file:
        data = json.load(json_file)

    small_chunk_size = len(data)//4
    chunk1 = data[:small_chunk_size]
    chunk2 = data[small_chunk_size:small_chunk_size*2]
    chunk3 = data[small_chunk_size*2:small_chunk_size*3]
    chunk4 = data[small_chunk_size*3:]
    chunks = [chunk1, chunk2, chunk3, chunk4]

    procs = []
    for i, chunk in enumerate(chunks):
        writer_p = Process(name="p_"+str(i),target=writerToQueue, args=((chunk, data_queue),))
        writer_p.start()
        procs.append(writer_p)
    for proc in procs:
        proc.join()

    formatted_data = []
    while not data_queue.empty():
        formatted_data.append(data_queue.get())
    with open("completedataV2.json", "w") as json_file:
        json.dump(formatted_data, json_file, indent=4)
    
