import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-layer Perceptron --> a class of feedforward ANN
class mlp_classifier(nn.Module):
    #num_classes is how many labels I have
    def __init__(self, input_size=63, hidden_size=128, num_classes=10):
        super(mlp_classifier, self).__init__()
        #nn.linear(in, out)
        self.fc1=nn.Linear(input_size, hidden_size)
        self.fc2=nn.Linear(hidden_size, hidden_size)
        self.fc3=nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x
