import torch
from pytorch_transformers import *
import json
import pprint
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
    for (sentence, index) in tokenized_sentences:
        input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) # tokenize the sentence
        outputs = model(input_ids) # compute
        final_layer = outputs[0].squeeze(0)
        print(final_layer.shape)
        representation = final_layer[0][index] # extract the representation of the word
        stack.append(representation)
    print("A peek at the representation of the words with the same definition:")
    print(stack[:3])
    compiled = torch.stack(stack)
    print("standard deviation of each entry:")
    stds = compiled.float().std(dim=0)
    print(stds)




"""
Convert the specified document from the json file to a list of sentences,
where each sentence is stored as a list of words.
"""
def getRawSentences(docname):
    with open("googledata.json") as json_file:
        data = json.load(json_file)

        sentences = []
        for document in data:
            if document["docname"] == docname or docname == "all":
                wordList = document["doc"]
                sentences = []
                sentence = []
                for x in range(len(wordList)):
                    
                    if wordList[x]["break_level"] == "SENTENCE_BREAK":
                        print(sentence)
                        sentences.append(sentence)
                        sentence = []
                    sentence.append(wordList[x]["text"])

    return sentences


"""

"""
def getSenses(docname):
    with open("googledata.json") as json_file:
        data = json.load(json_file)

        sentences = []
        for document in data:
            if document["docname"] == docname or docname == "all":
                wordList = document["doc"]
                sentences = []
                sentence = []
                cur_pos = -1
                for x in range(len(wordList)):
                    word = wordList[x]
                    
                    cur_pos += 1

                    if wordList[x]["break_level"] == "SENTENCE_BREAK":
                        print(sentence)
                        sentences.append(sentence)
                        sentence = []
                        cur_pos = 0
                    if "sense" in word:
                        sentence.append({"word": word["text"], "pos": cur_pos, "sense": word["sense"]})\
                            
    return sentences


"""
Accepts a single word in string format as a parameter. It then searches through
the specified document for all sentences containing that word. A list of lists is then created
where the inner list contains the relevant sentences and the position of the word of interest
in those sentences. This list of lists is then returned.
"""
def getTokenizedSentences(word, docname):
    with open("googledata.json") as json_file:
        data = json.load(json_file)

        exampleSentences = []

        for document in data:
            if document["docname"] == docname or docname == "all":
                wordList = document["doc"]
                exampleSentences.extend(makeSentences(wordList, word))
        return exampleSentences

def makeSentences(wordList, word):
    cur_pos = 0
    sentence = ""
    sentences = []
    word_pos = -1
    for x in range(len(wordList)):
        sentence += wordList[x]["text"]

        # Note: word_pos will be overwritten if the word occurs more than once
        if wordList[x]["text"] == word and "sense" in wordList[x]:
            word_pos = cur_pos
        cur_pos += 1
        if wordList[x]["text"] == ".":
            
            # only add the sentence if it contains the word of interest
            # and if that word has a sense
            if word_pos is not -1:
                sentences.append([sentence, word_pos])

            cur_pos = 0
            sentence = ""
            word_pos = -1
        else:
            sentence += " "
    return sentences


def trackRawSentenceIndices(bert_sent):
    tracking = []
    n = -1
    for bert_word in bert_sent:
        if not bert_word.startswith("##"):
            n += 1
        tracking.append(n)
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
    getSenses("/written/letters/112C-L014.txt")
    #compare_word_same_sense(sentences)

    