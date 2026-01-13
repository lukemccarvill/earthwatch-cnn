"""In progress - CNN PyTorch training script for a CNN to predict water 
quality based on citizen science images from Earthwatch - RT 09/01/26"""

#At the minute no validation, just training and test, need to divide the training data into training and validation. 

import torch #The core PyTorch library: tensors, autograd, device handling
import torch.nn as nn #Containts layers (convolutions, linear layers, pooling)
import torch.nn.functional as F #Contains operations/functions applied to data, like RELU
import torch.optim as optim #Contains optimisers like SGD, Adam, RMSprop
from tqdm import tqdm

from data_loader import get_dataloaders, classes #classes are tuple-like
trainloader, testloader = get_dataloaders() #Uses the data_loader.py file to return batches of (images, labels)
#Each batch: images[B, 3, 224, 224], labels:[B] (these are the dimensions, B is no. in the batch)

#Defines a neural network class Net that subclasses nn.Module
class Net(nn.Module):
    def __init__(self, num_classes=len(classes)): #No. labels
        super().__init__() #Initialises the base nn.Module class
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) #3 input channels (RGB), 16 output channels, 3x3 kernel, padding=1 to keep spatial size
        self.pool  = nn.MaxPool2d(2, 2)  #Halves spatial dimensions (height and width)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) #Deeper layers to learn more complex features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #Increase channels from 32 to 64. 

        #After 3 conv + pool (224 -> 112 -> 56 -> 28)
        #These are fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes) #This is the final classification layer

    def forward(self, x): #This defines how data moves through the network
        x = F.relu(self.conv1(x)); x = self.pool(x) #Convolution --> ReLU activation --> pooling
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x)); x = self.pool(x)
        x = torch.flatten(x, 1) #Converts [B, C, H, W] to [B, features] (flattens everythinig except batch dim)
        x = F.relu(self.fc1(x)) #Fully connected. 
        x = self.fc2(x) #This layer outputs raw logits, which are unnormalised, numerical scores produced by the final layer of a nn before being converted to probabilites
        return x

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Instantiate model, loss, optimizer
model = Net(num_classes=len(classes)).to(device) #Creates the CNN, moves to device
criterion = nn.CrossEntropyLoss() # Loss function for multi-class identification. Input: logits [B,C], target: integer labels [B]
optimizer = optim.Adam(model.parameters(), lr=1e-3) #Updates all model parameters

# Training loop SET THESE, NOT SURE
num_epochs = 10 #Full passes
print_every = 10  #Batches

#Epoch loop-each epoch is a full dataset pass
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(trainloader, 1):
        inputs = inputs.to(device) #Move data to the same device as the model
        targets = targets.to(device)
        optimizer.zero_grad() #Clears gradients from the previous step
        outputs = model(inputs)                #Shape (B, num_classes)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_every == 0:
            avg = running_loss / print_every
            print(f"Epoch {epoch}  Batch {i}  Loss {avg:.4f}")
            running_loss = 0.0

    #Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Epoch {epoch} finished — Test accuracy: {acc:.2f}%")

# Save the model weights
torch.save(model.state_dict(), "model_weights_1.pth")
print("Saved model_weights_1.pth")    
#Checkpoint
ckpt = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
classes = ckpt["classes"]

#To load and run the model you have to reacreate the model and then load in the weights. Takes up a lot less space than saving the whole model. 
#Code for this:
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = Net(num_classes=len(classes))
#model = model.to(device)
#state_dict = torch.load("model_simple.pth", map_location=device)
#model.load_state_dict(state_dict)
#model.eval()
#with torch.no_grad():
#outputs = model(inputs)
#preds = outputs.argmax(dim=1)

