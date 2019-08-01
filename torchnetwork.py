from __future__ import print_function, division
import torch
import numpy as np 
from compare import sampleData, sampleTrainingDataFromFile
from experiment import tensor_batcher
from train import train_net
from networks import SimpleClassifier

if torch.cuda.is_available():
    print("using gpu")
    cuda = torch.device('cuda:0')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        return model.cuda()
else: 
    print("using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        return model

trainingData, testData = sampleData(1000, 2100, 400)
trainingData = cudaify(trainingData)
testData = cudaify(testData)

classifier = cudaify(DropoutClassifier7(1536, 700,2))

train_net(classifier, trainingData, testData, tensor_batcher,
              batch_size=96, n_epochs=100, learning_rate=0.001,
              verbose=True)