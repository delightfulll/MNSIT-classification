import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import transforms
import torchvision.datasets as datasets


#declaring the actual neural network architecture
class Network(nn.Module):
    def __init__(self):
        #call the nn.module parent class
        super().__init__()
        #[120, 1, 28, 28] 28 * 28 * 1 = 784
        self.hidden0 = nn.Linear(784, 128)
        self.hidden1 = nn.Linear(128, 64)
        #we want 10 output features 0 - 9
        self.output = nn.Linear(64, 10)
        #using logsoftmax bc of nnnloss function
        self.softmax = nn.LogSoftmax(dim = 1)
        self.activation = nn.ReLU()
        #using dropout to avoid overfitting
        self.dropout = nn.Dropout(p = 0.30)


    #forward propagation run x through all layers
    def forward(self, x):
        #flatten first so dimentions match 784, then whatever the y dimention is
        x = x.view(x.shape[0], -1)
        #pass 1
        x = self.hidden0(x)
        x = self.activation(x)
        x = self.dropout(x)

        #pass 2
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.output(x)
        #apply softmax before returning bc we want probabilities for multi classification
        x = self.softmax(x)

        return x
    
#initialize the network
model = Network()
print(model)