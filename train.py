import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from gesture_models import mlp_classifier
from load_dataset import gesture_dataset

#step 1: Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 2

#step 2: Load dataset
train_data = gesture_dataset(csv_file='')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)

#step 3: Initialize model, loss function, optimizer
model = mlp_classifier(num_classes=len(train_data.label_map))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#step 4: Training loop
for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        #step 5: zero the gradients
        optimizer.zero_grad()

        #step 6: forward pass
        outputs = model(data)

        #step 7: calculate the loss
        loss = loss_fn(outputs, labels)

        #step 8: backpropagation
        loss.backward()

        #step 9: optimizer
        optimizer.step()

        running_loss += loss.item()

        #calculate accuracy
        _, predicted = torch.max(outputs,1) #get class with max score
        correct +=(predicted == labels).sum().item() #count correct predictions
        total += labels.size(0)

    #step 10: print statistics after each epoch
    epoch_loss = running_loss/len(train_loader)
    epoch_accuracy = correct/total *100
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

#step 11: save the model
torch.save(model.state_dict(), 'path')