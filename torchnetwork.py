from __future__ import print_function, division
import torch
import numpy as np 
from compare import sampleData, sampleFromFileTwoSenses, sampleDataTwoSenses, loadMostDiverseLemmas
from experiment import tensor_batcher
from train import train_net
from networks import SimpleClassifier, DropoutClassifier


def createAndTrainNN(file_name, trainingData, testData):

    if torch.cuda.is_available():
        print("using gpu")
        cuda = torch.device('cuda:2')
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor
        def cudaify(model):
            return model.cuda(cuda)
    else: 
        print("using cpu")
        cuda = torch.device('cpu')
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor
        def cudaify(model):
            return model

    #trainingData, testData = sampleFromFileTwoSenses(num_pairs, file_name, 0.8, senses)
    trainingData = cudaify(trainingData)
    testData = cudaify(testData)
    print("training size:", trainingData.shape)
    print("testing size:", testData.shape)
    print(file_name)


    classifier = cudaify(DropoutClassifier(1536, 100, 2))

    train_net(classifier, trainingData, testData, tensor_batcher,
                batch_size=96, n_epochs=10, learning_rate=0.001,
                verbose=True)

if __name__=="__main__":

    most_diverse_lemmas = loadMostDiverseLemmas()
    i = 1


    """
    most diverse lemmas contains a list of file_names sorted by the frequency of the 
    second most common sense. The loop below iterates through the lemmas in the specified
    position and trains a network on each. This process is extremely fast due to the
    small data size.
    """

    best_rank_to_allow = 0
    worst_rank_to_allow = 15


    for i in range(best_rank_to_allow, worst_rank_to_allow):
        file_name, num_pairs, sense1, sense2 = most_diverse_lemmas[i]
        if not num_pairs % 10 == 0:
            num_pairs -= num_pairs % 10
        trainingData, testData = sampleFromFileTwoSenses(100, file_name, 0.8, [sense1, sense2])

        createAndTrainNN(file_name, trainingData, testData)
        i += 1

    """
    The following loop gradually increases the number of files it is training on.
    """

    for i in range(3, 15, 3):
        files_to_read = most_diverse_lemmas[:i]
        #files_to_read = most_diverse_lemmas[i-3:i+3]
        pairs_train = []
        pairs_test = []
        all_files_string = ""
        for f in files_to_read:
            file_name, num_pairs, sense1, sense2 = f
            if not num_pairs % 10 == 0:
                num_pairs -= num_pairs % 10
            all_files_string += file_name+" "
            curr_train, curr_test = sampleFromFileTwoSenses(num_pairs, file_name, 0.8, [sense1, sense2])
            pairs_train.append(curr_train)
            pairs_test.append(curr_test)
        createAndTrainNN(all_files_string, torch.cat(pairs_train).float(), torch.cat(pairs_test).float())
        



    """     
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

    trainingData, testData = sampleDataTwoSenses(300, , .80)
    trainingData = cudaify(trainingData)
    testData = cudaify(testData)
    print("training size:", trainingData.shape)
    print("testing size:", testData.shape)


    classifier = cudaify(DropoutClassifier7(1536, 700, 2))

    train_net(classifier, trainingData, testData, tensor_batcher,
                batch_size=96, n_epochs=200, learning_rate=0.001,
                verbose=True) """