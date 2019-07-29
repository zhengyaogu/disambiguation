import torch
import pandas as pd
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
import random
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

def getLemmaFromWord():
    pass


def exploreData(file_):
    with open(file_) as f:
        file_data = json.load(f)

    id_sent_dict = {}
    with Cd("lemmadata"):
        for document in file_data:
            doc_body = document["doc"]
            for sent_id, sent_object in enumerate(doc_body):

                words_with_sense = sent_object["senses"]
                id_sent_dict[sent_id] = sent_object["bert_sent"]

                for word_object in words_with_sense:
                    lemma = getLemmaFromWord(word_object["word"])
                    lemma_instance = {}
                    lemma_instance["sent_id"] = sent_id
                    lemma_instance["pos"] = word_object["pos"]
                    lemma_instance["sense"] = word_object["sense"]
                    lemma_file_name = lemma+".json"
                    if os.path.exists(lemma_file_name):
                        with open(lemma_file_name, "r") as lemma_file:
                            lemma_instance_list = json.load(lemma_file)
                        lemma_instance_list.append(lemma_instance)
                        with open(lemma_file_name, "w") as lemma_file:
                            json.dump(lemma_instance_list)
                    else:
                        lemma_instance_list = [lemma_instance]
                        with open(lemma_file_name, "w") as lemma_file:
                            json.dump(lemma_instance_list, lemma_file)
            with open("id_to_sent.json", "w") as id_to_sent_file:
                json.dump(id_sent_dict, id_to_sent_file)



                

def createListOfUniqueWords(file_):
    with open(file_, "r") as f:
        file_data = json.load(f)
    unique_lemmas = []
    for document in file_data:
        doc_body = document["doc"]
        for word in doc_body:
            if "lemma" in word:
                if not word["lemma"] in unique_lemmas:
                    unique_lemmas.append(word["lemma"])
    print(len(unique_lemmas))
    with open("unique_lemmas.json", "w") as unique:
        json.dump(unique_lemmas, unique)





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
        print(word)
        instances = word_dict[word]
        pairs_of_word = []
        if len(instances) <= 1: continue
        for i in range(len(instances)):
            if i%1000 == 0:
                print(i)
            if i > 2000: break
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


def pairDataToBertVecs(train_size, instance_size, train_or_test):
    """
    return 2D torch tensor of 
    """
    with open("sense_pairs.json", "r") as f:
        comp_data = json.load(f)
        sent_dict = comp_data[0]
        word_pairs = comp_data[1].json
    
    # set up the model
    config = BertConfig.from_pretrained('bert-base-uncased')
    #config.output_hidden_states=True
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel(config)

    comb_vecs = []
    y_train = []
    k = 0
    random_keys_list = random.sample(word_pairs.keys(), 1000)
    for word in random_keys_list:
        if k >= train_size: break
        print(word)
        word_pairs_of_word = word_pairs[word]
        i = 0
        for pair in word_pairs_of_word:
            if i >= 10: break
            instance1 = pair[0]
            instance2 = pair[1]
            if_same = pair[2]

            sent1 = sent_dict[str(instance1[0])]
            pos1 = instance1[1]
            sent2 = sent_dict[str(instance2[0])]
            pos2 = instance2[1]

            vec1 = vectorizeWordInContext(sent1, pos1, tokenizer, model)
            vec2 = vectorizeWordInContext(sent2, pos2, tokenizer, model)

            comb_vec = torch.cat((vec1, vec2))
            comb_vecs.append(comb_vec)

            y_train.append(if_same)
            i += 1
        k += 1
    x_train = torch.stack(comb_vecs)
    if train_or_test == "train":
        torch.save([x_train, y_train], "logistics_train.json")
        print("dump train success!")
    elif train_or_test == "test":
        torch.save([x_train, y_train], "logistics_test.json")
        print("dump test success!")
    return x_train


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
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *0.9, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def generateWordLemmaDict():
    with open("googledata.json", "r") as f:
        data = json.load(f)
    d = {}
    for doc in data:
        print("processing", doc["docname"], "...")
        for word in doc["doc"]:
            if "lemma" in word.keys() and "sense" in word.keys():
                d[word["text"]] = word["lemma"]
    with open("word_lemma_dict.json", "w") as f:
        json.dump(d, f, indent=4)

def getLemmaFromWord(word):
    with open("word_lemma_dict.json", "r") as f:
        d = json.load(f)
    return d[word]



if __name__ == "__main__":
    memory_limit() # Limitates maximun memory usage to half
    #allWordPairs("completedata.json")
    #pairDataToBertVecsFiles(2000)
    #loadBertVecTrainingData()
    #print(unicodeToAscii("fam\u00adily"))
    #documents_to_process = test_files_list
    #formatted_data = getFormattedData(["all"])
    #with open("completedata.json", "w") as json_file:
    #json.dump(formatted_data, json_file, indent=4)
    createListOfUniqueWords("googledata.json")
    