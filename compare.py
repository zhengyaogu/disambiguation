import torch
from pytorch_transformers import *
import json
import pprint
import copy
"""
config = BertConfig.from_pretrained('bert-base-uncased')
config.output_hidden_states=True
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification(config)
input_ids = torch.tensor(tokenizer.encode("I am a student")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
final_layer = outputs[1][-1]
"""
def compare_word_same_sense(tokenized_sentences):
    # set up the model
    config = BertConfig.from_pretrained('bert-base-uncased')
    #config.output_hidden_states=True
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel(config)
    # iterate through sentences and extract the representation of the word
    stack = []
    for sentence_pack in tokenized_sentences:
        bert_sent = sentence_pack["bert_sent"]

        
    print("A peek at the representation of the words with the same definition:")
    print(stack[:3])
    compiled = torch.stack(stack)
    print("standard deviation of each entry:")
    stds = compiled.float().std(dim=0)
    print(stds)


def BreakToString(break_level):
    if break_level == "NO_BREAK" or break_level == "SENTENCE_BREAK":
        return ""
    else:
        return " "

# Makes an english readable sentence from a list of word objects(
# dictionaries from the json file). Returns the sentence as a string.
def makeSentence(json_sent):
    sent = ""
    for word in json_sent:
        sent += BreakToString(word["break_level"]) + word["text"]
    return sent

def makeRawSentence(json_sent):
    sent = []
    for word in json_sent:
        sent.append(word["text"])
    return sent

def makeSenseSentence(json_sent):
    senses = []
    cur_pos = 0
    for word in json_sent:
        if "sense" in word:
            senses.append({"word": word["text"], "pos": cur_pos, "sense": word["sense"]})
        cur_pos += 1
    return senses      

def getJsonSentences(data):
    sentence = []
    sentences = []
    for word in data:
        if word["break_level"] == "SENTENCE_BREAK":
            sentences.append(sentence)
            sentence = []
        sentence.append(word)
    return sentences

def getFormattedData(docnames):
    data = None
    with open("googledata.json") as json_file:
        data = json.load(json_file)

    formatted_data = []
    for document in data:
        if document["docname"] in docnames or "all" in docnames:
            doc_dict = {}
            doc_dict["docname"] = document["docname"]
            doc_dict["doc"] = []
            createSentenceDictionaries(document["doc"], doc_dict["doc"])
            formatted_data.append(doc_dict)
            print("finished processing the document: " + document["docname"])
    return formatted_data
    

def createSentenceDictionaries(document_data, list_to_modify):
    for sent in getJsonSentences(document_data):
        natural_sent = ""
        raw_sent = []
        senses = []
        cur_pos = 0
        for word in sent:
            raw_sent.append(word["text"])
            natural_sent += BreakToString(word["break_level"]) + word["text"]
            if "sense" in word:
                senses.append({"word": word["text"], "pos": cur_pos, "sense": word["sense"]})
            cur_pos += 1
        bert_sent = getBertSentenceFromRaw(raw_sent)
        tracking = trackRawSentenceIndices(raw_sent, bert_sent)

        sent_dict = {}
        sent_dict["natural_sent"] = natural_sent
        sent_dict["sent"] = raw_sent
        sent_dict["bert_sent"] = bert_sent
        sent_dict["tracking"] = tracking
        sent_dict["senses"] = senses
        list_to_modify.append(sent_dict)
    return sent_dict


def trackRawSentenceIndices(raw_sent, bert_sent):
    tracking = []
    n = 0
    i = 0 #keep track of the current word in bert_sent
    for raw_word in raw_sent:
        while i < len(bert_sent):
            if raw_word == "":
                break
            curr_bert = bert_sent[i]
            if curr_bert.startswith("##"):
                curr_bert = curr_bert[2:]
            tracking.append(n)
            raw_word = raw_word[len(curr_bert):]
            i += 1
        n += 1
    return tracking


def getBertSentenceFromRaw(raw_sent):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_sent_list = []
    for raw_word in raw_sent:
        bert_tokens = tokenizer.tokenize(raw_word)
        bert_sent_list += bert_tokens
    return bert_sent_list

def getSentencesBySense(sense):
    filtered_sentences = []
    with open("formattedgoogledata.json", "r") as f:
        data = json.load(f)
        for doc in data:
            for sentence in doc["doc"]:
                for word_with_sense in sentence["senses"]:
                    if word_with_sense["sense"] == sense:
                        filtered_sentences.append(sentence)
        return filtered_sentences

def getSentencesByWord(word):
    filtered_sentences = []
    with open("formattedgoogledata.json", "r") as f:
        data = json.load(f)
        for doc in data:
            for sentence in doc["doc"]:
                if word in sentence["sent"]:
                    filtered_sentences.append(sentence)
    return filtered_sentences

def allWordPairs():
    tk = BertTokenizer.from_pretrained('bert-base-uncased')
    word_dict = {}
    with open("formattedgoogledata3.json", "r") as f:
        data = json.load(f)
        for doc in data:
            for sentence in doc["doc"]:
                for word in sentence["senses"]:
                    vocab = word["word"]
                    tokenized_vocab = tk.tokenize(vocab)
                    if len(tokenized_vocab) > 1: continue
                    raw_sent = sentence["sent"]
                    bert_sent = sentence["bert_sent"]
                    raw_pos = word["pos"]
                    tracking = sentence["tracking"]
                    if not vocab in word_dict.keys():
                        word_dict[vocab] = []
                    pos = tracking.index(raw_pos)
                    word_dict[vocab].append([bert_sent, pos, word["sense"]])
    pairs = {}
    for word in word_dict.keys():
        instances = word_dict[word]
        pairs_of_word = []
        if len(instances) <= 1: continue
        for i in range(len(instances)):
            j = i + 1
            while j < len(instances):
                same_sense = 1 if instances[i][2] == instances[j][2] else 0
                pairs_of_word.append([instances[i][:2], instances[j][:2], same_sense])
                j += 1
        pairs[word] = pairs_of_word
    print(pairs)






"""
if __name__ == "__main__":
    documents_to_process = ["all"]
    formatted_data = getFormattedData(documents_to_process)
    with open("formattedgoogledata3.json", "w") as json_file:
        json.dump(formatted_data, json_file, indent=4)
"""
    