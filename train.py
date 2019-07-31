from torch import optim
import torch
import sys



def evaluate(net, dev, batcher):
    """
    Evaluates a trained neural network classifier on a partition of a
    data manager (e.g. "train", "dev", "test").

    The accuracy (i.e. percentage of correct classifications) is
    returned, along with a list of the misclassifications. Each
    misclassification is a triple (phrase, guessed, actual), where
        - phrase is the misclassified phrase
        - guessed is the classification made by the classifier
        - actual is the correct classification
    
    """    
    def accuracy(outputs, labels):
        correct = 0
        total = 0
        misclassified = []
        for (i, output) in enumerate(outputs):
            total += 1
            if labels[i] == output.argmax():
                correct += 1            
        return correct, total, misclassified
    val_loader = batcher(dev, 128)
    total_val_loss = 0
    correct = 0
    total = 0
    misclassified = []
    loss = torch.nn.CrossEntropyLoss()    
    for data in val_loader:
        inputs = data[:,1:]
        labels = torch.clamp(data[:,0], min=0).long()

        val_outputs = net(inputs)            
        val_loss_size = loss(val_outputs, labels)

        correct_inc, total_inc, misclassified_inc = accuracy(val_outputs, 
                                                             labels)
        correct += correct_inc
        total += total_inc
        misclassified += misclassified_inc
        total_val_loss += val_loss_size.data.item()
    return correct/total, misclassified       


def train_net(net, train, dev, batcher, batch_size, n_epochs, learning_rate, verbose=True):
    """
    Trains a neural network classifier on the 'train' partition of the
    provided DataManager.
    
    The return value is the trained neural network.
    
    """    
    def log(text):
        if verbose:
            sys.stdout.write(text)
                
    loss = torch.nn.CrossEntropyLoss()    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)    
    best_net = net
    best_acc = 0.0
    log("Training classifier.\n")
    for epoch in range(n_epochs):      
        train_loader = batcher(train, batch_size)
        log("  Epoch {} Accuracy = ".format(epoch))
        running_loss = 0.0
        total_train_loss = 0       
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs = data[:,1:]
            labels = torch.clamp(data[:,0], min=0).long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()             
        net.eval()
        acc, misclassified = evaluate(net, dev, batcher)
        if acc > best_acc:
            best_net = net
            best_acc = acc
        log("{:.2f}\n".format(acc))
    return best_net

