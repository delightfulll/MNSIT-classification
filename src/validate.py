import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import model
from datasets import train_dataloader, val_dataloader, test_dataloader
from train import loss_func, val_loss, train_loss, min_val_loss, device, epoch_train_loss, epoch_val_loss


#set the model to eval mode
model.eval()

#get rid of gradients during validation phase
with torch.no_grad():
        for image, label in val_dataloader:
            #set it again
            image, label = image.to(device), label.to(device)
            output = model(image)
            
            #loss function used to compare
            loss = loss_func(output, label)
            
            #add to the epoch_val_loss
            epoch_val_loss += loss.item()


    #calculate total loss per epoch
epoch_val_loss = epoch_val_loss / len(val_dataloader)
epoch_train_loss = epoch_train_loss / len(train_dataloader)
    #append it to history so we can keep track of it
val_loss.append(epoch_val_loss)
train_loss.append(epoch_train_loss)

model.train()