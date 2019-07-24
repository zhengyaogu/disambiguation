import torch
from pytorch_transformers import *
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