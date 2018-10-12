# Importing Modules

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import os
import numpy as np
from matplotlib import pyplot as plt

# Loading Data

def find_data_dir():
    data_path = 'data'
    while os.path.exists(data_path) != True:
        data_path = '../' + data_path
        
    return data_path

# MNIST dataset
mnist_train = datasets.MNIST(root=find_data_dir(),
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
print("Downloading Train Data Done ! ")

mnist_test = datasets.MNIST(root=find_data_dir(),
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
print("Downloading Test Data Done ! ")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# our model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)
    
    def forward(self, X):
        X = F.relu((self.linear1(X)))
        X = self.linear2(X)
        return X

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Phase

batch_size = 100

data_iter = DataLoader(mnist_train, batch_size=100, shuffle=True, num_workers=1)

print("Iteration maker Done !")

# Training loop
for epoch in range(10):
    avg_loss = 0
    total_batch = len(mnist_train) // batch_size
    for i, (batch_img, batch_lab) in enumerate(data_iter):
        X = batch_img.view(-1, 28*28).to(device)
        Y = batch_lab.to(device)

        y_pred = model.forward(X)

        loss = criterion(y_pred, Y)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
        if (i+1)%100 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, " Loss : ", avg_loss.cpu().data.numpy()/(i+1))
    print("Epoch : ", epoch+1, " Loss : ", avg_loss.cpu().data.numpy()/total_batch)
print("Training Done !")

# Evaluation

test_img = mnist_test.test_data.view(-1, 28*28).float().to(device)
test_lab = mnist_test.test_labels.to(device)
outputs = model.forward(test_img)
pred_val, pred_idx = torch.max(outputs.data, 1)
correct = (pred_idx == test_lab).sum()
print('Accuracy : ', correct.data.cpu().numpy()/len(test_img)*100)

# Testing

r = np.random.randint(0, len(mnist_test)-1)
X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28*28).float().to(device)
Y_single_data = mnist_test.test_labels[r:r + 1]

single_prediction = model(X_single_data)
plt.imshow(X_single_data.data.view(28,28).numpy(), cmap='gray')
plt.title("Label : {}, Prediction : {}".format(Y_single_data.data.cpu().view(1).numpy(), torch.max(single_prediction.data, 1)[1].numpy()))
plt.show()