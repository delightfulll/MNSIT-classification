import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import model
from datasets import train_dataloader, val_dataloader, test_dataloader
from train import loss_func, val_loss, train_loss, min_val_loss, device, epoch_train_loss, epoch_val_loss



#testing our model on test data
model.eval()
test_loss = 0.0
test_acc = 0.0

#dictionary to keep track of correctness from model
correct_count = {i:0 for i in range(10)}
total_count = {i:0 for i in range(10)}

#store predictions and actual labels for visualization
#we'll just grab one batch for visualization - simpler!
sample_images = None
sample_labels = None
sample_predictions = None

with torch.no_grad():
    for images, labels in test_dataloader:
        #move to gpu if exists
        images, labels = images.to(device), labels.to(device)
        #make prediction
        outputs = model(images)
        #since using log softmax, convert each output to exp
        pred_label_proba = torch.exp(outputs)
        
        loss = loss_func(outputs, labels)
        #compute test loss
        test_loss += loss.item()
        
        #get predicted labels (indices of max values)
        #torch.max returns (max_values, indices) - we only want indices, so use _
        _, pred_labels = torch.max(pred_label_proba, dim=1)
        
        #calculate accuracy for this batch
        correct = (pred_labels == labels).sum().item()
        test_acc += correct
        
        #save first batch for visualization (much simpler!)
        if sample_images is None:
            sample_images = images.cpu()  #keep as tensor, convert to numpy later
            sample_labels = labels.cpu()
            sample_predictions = pred_labels.cpu()
        
        #update per-class statistics
        for i in range(len(labels)):
            label = labels[i].item()  #convert tensor to regular number
            total_count[label] += 1
            if pred_labels[i] == labels[i]:
                correct_count[label] += 1

#calculate average test loss and accuracy
test_loss = test_loss / len(test_dataloader)
test_acc = test_acc / len(test_dataloader.dataset)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
print('\nPer-class accuracy:')
for i in range(10):
    if total_count[i] > 0:
        acc = correct_count[i] / total_count[i]
        print(f'Class {i}: {acc:.4f} ({acc*100:.2f}%) - {correct_count[i]}/{total_count[i]} correct')
