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

"""
Accepts a single word in string format as a parameter. It then searches through
the specified document for all sentences containing that word. A list of lists is then created
where the inner list contains the relevant sentences and the position of the word of interest
in those sentences. This list of lists is then returned.
"""
def getTokenizedSentences(docname):
    with open("googledata.json") as json_file:
        data = json.load(json_file)

        exampleSentences = []

        for document in data:
            if document["docname"] == docname or docname == "all":
                wordList = document["doc"]
                exampleSentences.extend(makeSentences(wordList))
        return exampleSentences


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

def makeBertSentence(raw_sent):
    return getBertSentenceFromRaw(raw_sent)

def makeTracking(bert_sent):
    return trackRawSentenceIndices(bert_sent)

def getJsonSentences(data):
    sentence = []
    sentences = []
    for word in data:
        if word["break_level"] == "SENTENCE_BREAK":
            sentences.append(sentence)
            sentence = []
        sentence.append(word)
    return sentences

def getFormattedData(docname):
    data = None
    relevant_data = None
    with open("googledata.json") as json_file:
        data = json.load(json_file)

    relevant_data = []
    for document in data:
        if document["docname"] == docname or docname == "all":
            relevant_data.extend(document["doc"])

    formatted_data = []
    for sent in getJsonSentences(relevant_data):
        sent_dict = {}
        sent_dict["natural_sent"] = makeSentence(sent)
        sent_dict["sent"] = makeRawSentence(sent)
        sent_dict["bert_sent"] = makeBertSentence(sent_dict["sent"])
        sent_dict["tracking"] = makeTracking(sent_dict["bert_sent"])
        sent_dict["senses"] = makeSenseSentence(sent)
        formatted_data.append(sent_dict)
    return formatted_data


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
        for sentence in data:
            for word_with_sense in sentence["senses"]:
                if word_with_sense["sense"] == sense:
                    new_sentence = copy.copy(sentence)
                    new_sentence["pos"] = word_with_sense["pos"]
                    filtered_sentences.append(sentence)
    return filtered_sentences

def getSentencesByWord(word):
    filtered_sentences = []
    with open("formattedgoogledata.json", "r") as f:
        data = json.load(f)
        for sentence in data:
            if word in sentence["sent"]:
                filtered_sentences.append(sentence)
    return filtered_sentences







if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    
    #sentences = getTokenizedSentences("letters", "all")
    #pp.pprint()
    pp.pprint(getFormattedData("/written/letters/112C-L014.txt"))
    
    #compare_word_same_sense(sentences)

    