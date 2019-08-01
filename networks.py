# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module): 
    """
    A simple neural network with a single ReLU activation
    between two linear layers.
    
    Softmax is applied to the final layer to get a (log) probability
    vector over the possible labels.
    
    """    
    def __init__(self, input_size, hidden_size, num_labels):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = self.linear1(input_vec)
        nextout = nextout.clamp(min=0)        
        nextout = self.linear2(nextout)
        return F.log_softmax(nextout, dim=1)

class DropoutClassifier(nn.Module): 
 
    def __init__(self, input_size, hidden_size, num_labels):
        super(DropoutClassifier, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.dropout1(nextout)
        nextout = self.linear1(nextout)
        nextout = nextout.clamp(min=0)
        #nextout = self.linear2(nextout).clamp(min=0)
        nextout = self.dropout2(nextout)    
        nextout = self.linear3(nextout)
        return F.log_softmax(nextout, dim=1)

class DropoutClassifier7(nn.Module):

    def __init__(self, input_size, hidden_size, num_labels):
        super(DropoutClassifier7, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=0.2)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.dropout4 = nn.Dropout(p=0.2)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.dropout5 = nn.Dropout(p=0.2)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.dropout6 = nn.Dropout(p=0.2)
        self.linear7 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.dropout1(nextout)
        nextout = self.linear1(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self.dropout2(nextout)
        nextout = self.linear2(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self.dropout3(nextout)
        nextout = self.linear3(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self.dropout4(nextout)
        nextout = self.linear4(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self.dropout5(nextout)
        nextout = self.linear5(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self.dropout6(nextout)
        nextout = self.linear6(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self.linear7(nextout)
        return F.log_softmax(nextout, dim=1)


