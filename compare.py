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
	config.output_hidden_states=True
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertForSequenceClassification(config)
	# iterate through sentences and extract the representation of the word
	stack = []
	for sentence in tokenized_sentences:
		index = sentence[1] # the position the word is at
		sentence = sentence[0] # the sentence string
		input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0) # tokenize the sentence
		outputs = model(input_ids) # compute
		final_layer = outputs[1][-1]
		representation = final_layer[0][index] # extract the representation of the word
		stack.append(representation)
	print("A peek at the representation of the words with the same definition:")
	print(stack[:3])
	compiled = torch.stack(stack)
	print("standard deviation of each entry:")
	stds = compiled.float().std(dim=0)
	print(stds)


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

        
    


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    
    sentences = getTokenizedSentences("letters", "all")
    pp.pprint(sentences)

    #compare_word_same_sense(sentences)

    