import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from gesture_models import mlp_classifier
from load_dataset import gesture_dataset

#step 1: Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 200

#step 2: split dataset and create dataloader
dataset = gesture_dataset(csv_file='hand_landmarks.csv')
#split 80 train, 20 test
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
train_data, test_data = random_split(dataset,[train_size, test_size])

#data loader 
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = False)

#step 3: Initialize model, loss function, optimizer
model = mlp_classifier(num_classes=len(dataset.label_map))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#step 4: Training loop
for epoch in range(epochs):
    #change model to training mode (otherwise it will be in inference mode)
    model.train()

    train_running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    #data is split into batches
    for batch_idx, (data, labels) in enumerate(train_loader):
        #step 5: zero the gradients
        optimizer.zero_grad()

        #step 6: forward pass (feeds input through the model)
        train_outputs = model(data)

        #step 7: calculate the loss
        train_loss = loss_fn(train_outputs, labels)

        #step 8: backpropagation
        train_loss.backward()

        #step 9: updates model weights using computed gradients
        optimizer.step()

        #accumulate the total loss during an epoch
        train_running_loss += train_loss.item()

        #calculate accuracy
        #get index of max score within a class
        _, predicted = torch.max(train_outputs,1) 
        #count correct predictions
        train_correct +=(predicted == labels).sum().item() 
        #keeps track of how many predictions have been made so far
        train_total += labels.size(0)

    #average loss per batch across an epoch for train data
    train_loss = train_running_loss/len(train_loader)
    train_accuracy = train_correct/train_total *100

    #step 10: model evaluation
    model.eval()
    test_running_loss=0.0
    test_correct = 0
    test_total=0
    #test data do not need gradient
    with torch.no_grad():
        for data, labels in test_loader:
            test_outputs = model(data)

            test_loss = loss_fn(test_outputs, labels)
            test_running_loss += test_loss.item()

            _,predicted=torch.max(test_outputs,1)
            test_correct +=(predicted==labels).sum().item()
            test_total += labels.size(0)
    
    test_loss = test_running_loss/len(test_loader)
    test_accuracy = test_correct/test_total*100



    #print statistics for train and test
    print(f"Epoch {epoch+1}/{epochs},train_Loss: {train_loss:.4f},train_Accuracy: {train_accuracy:.2f}%,test_loss: {test_loss:.4f},test_Accuracy: {test_accuracy:.2f}%")

#step 11: save the model
torch.save(model.state_dict(), 'D:\Desktop\Machine-Learning\gesture\gesture_mlp.pt')