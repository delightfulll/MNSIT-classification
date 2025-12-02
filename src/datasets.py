import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import transforms
import torchvision.datasets as datasets



#define a list of transforms because we might use more later, can also resize
transform = transforms.Compose([transforms.ToTensor()])

#convet the data into tensors using transform arg

#define two sets, a training set and validation (test) set
train_ds = datasets.MNIST(root = 'data', train = True, transform = transform , download = True)
test_ds = datasets.MNIST(root = 'data', train = False, transform = transform , download = True)


print(train_ds, test_ds)

#split the dataset into 80% training and 20% test (validation) set
train_num = len(train_ds)

#make a list of the indicies
indicies = list(range(len(train_ds)))
#shuffle the indicies
np.random.shuffle(indicies)

#split the inidicies into 20 percent use floor to get rid of decimals and make into whole int
split = int(np.floor(0.2 * train_num))

#retrieve the data
val_indx, train_indx = indicies[:split], indicies[split:]


#dataloader uses sampler to pass in because we want to shuffle from the train indexes and test indexes by themselves

#sampler randomly samples from a list of given indexes
train_sampler = sampler.SubsetRandomSampler(train_indx)
val_sampler = sampler.SubsetRandomSampler(val_indx)


#create the train, test, validation dataloaders
train_dataloader = DataLoader(train_ds, batch_size = 32, sampler = train_sampler)

#for updating hyperparameters
val_dataloader = DataLoader(train_ds, batch_size = 32, sampler = val_sampler)

#don't shuffle test_dataloader because order doesn't matter
test_dataloader = DataLoader(test_ds, batch_size = 32)