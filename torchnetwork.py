from __future__ import print_function, division
import torch
import numpy as np 
from compare import sampleAllTrainingData, sampleTrainingDataFromFile
from experiment import tensor_batcher
from train import train_net
from networks import SimpleClassifier
trainingDataTensor = sampleAllTrainingData(100, 2000)


numExamples = len(trainingDataTensor)
permutation = torch.randperm(numExamples)
trainingDataTensor = trainingDataTensor[permutation]
trainingData, testData = trainingDataTensor[:int(numExamples*.8)], trainingDataTensor[int(numExamples*.2):]

trainLoader = tensor_batcher(trainingDataTensor, 1000)

classifier = SimpleClassifier(1536, 96,2)
train_net(classifier, trainingData, testData, tensor_batcher,
              batch_size=96, n_epochs=30, learning_rate=0.001,
              verbose=True)