import torch
from pytorch_transformers import *
import json
import pprint
import copy
import timeit
import unicodedata
import resource
import sys
import os
import string
from cd import Cd


training_files_list = ["/written/letters/112C-L014.txt", 
                      "/written/blog/Acephalous-Internet.txt",
                      "/written/email/lists-003-2144868.txt",
                      "/written/essays/Madame_White_Snake.txt",
                      "/written/ficlets/1403.txt",
                      "/written/fiction/A_Wasted_Day.txt",
                      "/written/govt-docs/chapter-10.txt",
                      "/written/jokes/jokes7.txt",
                      "/written/journal/Article247_327.txt",
                      "/written/movie-script/JurassicParkIV-Scene_3.txt",
                      "/written/newspaper:newswire/NYTnewswire9.txt",
                      "/written/non-fiction/chZ.txt",
                      "/written/spam/111410.txt",
                      "/written/technical/1471-230X-2-21.txt",
                      "/written/travel-guides/HistoryGreek.txt",
                      "/written/twitter/tweets1.txt",
                      "/spoken/face-to-face/RindnerBonnie.txt",
                      "/spoken/telephone/sw2015-ms98-a-trans.txt"
                      ]
test_files_list = ["/spoken/debate-transcript/2nd_Gore-Bush.txt"]



def compare_word_same_sense(tokenized_sentence):
    # set up the model
    config = BertConfig.from_pretrained('bert-base-uncased')
    #config.output_hidden_states=True
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel(config)

    sentence = tokenized_sentence[0]
    pos = tokenized_sentence[1]

    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(sentence)).unsqueeze(0)
    outputs = model(input_ids)
    final_layer = outputs[0].squeeze(0)
    word_representation = final_layer[pos]
    print(word_representation)

def vectorizeWordInContext(sentence, pos, tokenizer, model):
    """
    take a word and its bert-style tokenized sentence to compute the vectorization in the context of a sentence.
    return the vector representation of the word
    """
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(sentence)).unsqueeze(0)
    outputs = model(input_ids)
    final_layer = outputs[0].squeeze(0)
    return final_layer[pos]


def BreakToString(break_level):
    """
    This helper function returns the correct string to use when creating the
    natural sentence, based on the specified break level.
    """
    if break_level == "NO_BREAK" or break_level == "SENTENCE_BREAK":
        return ""
    else:
        return " "


def makeSentence(json_sent):
    """
    Note: the make*(json_sent) functions are no longer called
    This function uses the json sentence to create an english readable 
    sentence. This sentence is referred to elsewhere as natural_sent.
    """
    sent = ""
    for word in json_sent:
        sent += BreakToString(word["break_level"]) + word["text"]
    return sent



def makeRawSentence(json_sent):
    """
    This function uses the json sentence to create a list of the words
    contained in the json sentence. This list is referred to elsewhere
    as raw_sent.
    """
    sent = []
    for word in json_sent:
        sent.append(word["text"])
    return sent


def makeSenseSentence(json_sent):
    """
    This function uses the json sentence to create a list of dictionaries.
    A dictionary is only created for words that have a sense attribute in the
    json sentence. The output of this function is referred to elsewhere as 
    senses.
    """
    senses = []
    cur_pos = 0
    for word in json_sent:
        if "sense" in word:
            senses.append({"word": word["text"], "pos": cur_pos, "sense": word["sense"]})
        cur_pos += 1
    return senses      

def getJsonSentences(data):
    """
    This function processes a single documents data. It scans through the words
    until it finds a sentence break and uses this to return a list of all the sentences
    in the document, where each sentence is a list of words.
    """
    sentence = []
    sentences = []
    for word in data:
        word["text"] = unicodeToAscii(word["text"])
        if word["break_level"] == "SENTENCE_BREAK":
            sentences.append(sentence)
            sentence = []
        if word["text"] != "":
            sentence.append(word)
    return sentences

def getFormattedData(docnames):
    """
    This function converts all the specified documents into our second json format.
    """
    data = []
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
    """
    This helper function aids getFormattedData by creating the 'sentence' dictionaries
    and appending them to the sent formatted_data list.
    """
    sent_dict = {}
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
    """
    track the position each word in BERT tokenization belongs to in the original tokenization
    returns the tracking list
    """
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
    """
    convert the original tokenization to the BERT tokenization
    returns the BERT tokenization
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_sent_list = []
    for raw_word in raw_sent:
        bert_tokens = tokenizer.tokenize(raw_word)
        bert_sent_list += bert_tokens
    return bert_sent_list

def getSentencesBySense(sense):
    """
    return sentences with word with a specific sense
    """
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
    """
    return sentences with specific word
    """
    filtered_sentences = []
    with open("formattedgoogledata.json", "r") as f:
        data = json.load(f)
        for doc in data:
            for sentence in doc["doc"]:
                if word in sentence["sent"]:
                    filtered_sentences.append(sentence)
    return filtered_sentences

def allWordPairs(filename):
    "return all the word pairs in a file, compaired in senses"
    tk = BertTokenizer.from_pretrained('bert-base-uncased')
    word_dict = {}
    sent_dict = {}
    sent_key = 0
    with open(filename, "r") as f:
        data = json.load(f)
    for doc in data:
        print("converting data in", doc["docname"])
        for sentence in doc["doc"]:
            sent_dict[sent_key] = sentence["bert_sent"]
            for word in sentence["senses"]:
                vocab = word["word"].lower()
                tokenized_vocab = tk.tokenize(vocab)
                if len(tokenized_vocab) > 1: continue
                raw_pos = word["pos"]
                tracking = sentence["tracking"]
                if not vocab in word_dict.keys():
                    word_dict[vocab] = []
                pos = tracking.index(raw_pos)
                word_dict[vocab].append([sent_key, pos, word["sense"]])
            sent_key += 1
    data = None
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
    with open("complete_sense_pairs.json", "w") as pair_file:
        json.dump([sent_dict, pairs], pair_file, indent = 4)



def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1 # Plus EOS marker
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

"""
Creates a wordData directory, which has subdirectories for each word unique word
contained in sense_pairs.json. Each of these directories contains one file for each
training example. 
"""
def pairDataToBertVecsFiles(limit_combos=100000):
    with open("sense_pairs.json", "r") as f:
        comp_data = json.load(f)
        sent_dict = comp_data[0]
        word_pairs = comp_data[1]
    
    # set up the model
    config = BertConfig.from_pretrained('bert-base-uncased')
    #config.output_hidden_states=True
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel(config)
    with Cd("wordData"):
        for word in word_pairs.keys():
            with Cd(word):
                print(os.getcwd())
                word_combinations = word_pairs[word]
                print(word, "has", len(word_combinations), "combinations")
                print("however only", limit_combos, "will be processed")
                for i, comb in enumerate(word_combinations):
                    if i >= limit_combos:
                        break
                    instance1 = comb[0]
                    instance2 = comb[1]
                    is_same_sense = comb[2]

                    sent1 = sent_dict[str(instance1[0])]
                    pos1 = instance1[1]
                    sent2 = sent_dict[str(instance2[0])]
                    pos2 = instance2[1]

                    vec1 = vectorizeWordInContext(sent1, pos1, tokenizer, model)
                    vec2 = vectorizeWordInContext(sent2, pos2, tokenizer, model)

                    comb_vec = torch.cat((vec1, vec2))
                    data_point = [comb_vec.tolist(), is_same_sense]

                    with open(word+str(i)+".json", "w") as f:
                        json.dump(data_point, f, indent=4)


def loadBertVecTrainingData():
    num_same = 0
    num_diff = 0
    X = []
    y = []
    with Cd("wordData"):
        for f in os.listdir():
            if os.path.isdir(f):
                print(f)
                with Cd(f):
                    for file_num, fil in enumerate(os.listdir()):
                        if file_num > 500:
                            continue
                        if os.path.isfile(fil):
                            data_point = None
                            with open(fil, "r") as d:
                                data_point = json.load(d)
                            if data_point[1] == 1 and num_same-num_diff < 25:
                                X.append(data_point[0])
                                y.append(data_point[1])
                                num_same += 1
                            elif data_point[1] == 0 and num_diff-num_same < 25:
                                X.append(data_point[0])
                                y.append(data_point[1])
                                num_diff += 1
    print(num_same)
    print(num_diff)
    print(len(X))
    print(len(y))
    print("writing trainingx.json")
    with open("trainingx.json", "w") as xfile:
        json.dump(X, xfile)
    print("writing trainingy.json")
    with open("trainingy.json", "w") as yfile:
        json.dump(y, yfile)
    


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *0.75, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

if __name__ == "__main__":
    memory_limit() # Limitates maximun memory usage to half
    allWordPairs("completedata.json")
    #pairDataToBertVecs(1000)
    #loadBertVecTrainingData()
    #print(unicodeToAscii("fam\u00adily"))
    #documents_to_process = test_files_list
    #formatted_data = getFormattedData(["all"])
    #with open("completedata.json", "w") as json_file:
    #json.dump(formatted_data, json_file, indent=4)
    