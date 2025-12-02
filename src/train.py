import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import model
from datasets import train_dataloader, val_dataloader, test_dataloader

#check if cuda exists and what device is being used
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print('using', device, 'for training')


#declare loss and optimizer functions
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

#declare epoches
n_epoches = 10 
#keep track of the losses
val_loss = []
#keep track of the losses
train_loss = []
min_val_loss = np.inf

#start test training
for epoch in range(1, n_epoches + 1):
    #keep track of the losses for each epoch
    epoch_val_loss = 0
    epoch_train_loss = 0

    #TRAINING LOOP

    #set model to train mode
    model.train()
    #for each batch
    for image, label in train_dataloader:

        #move the label and image to the same device as model if available
        image, label = image.to(device), label.to(device) 
        #calculate the prediciton
        output = model(image)
        #loss function
        loss = loss_func(output, label)
        #zero the grad before updating grads
        optimizer.zero_grad()
        #update grad
        loss.backward()
        #update the grads from backprop
        optimizer.step()
        #add loss (the value, not the actual tensor itself)
        epoch_train_loss += loss.item()
