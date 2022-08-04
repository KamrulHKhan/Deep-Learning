# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:14:59 2022

@author: Md Kamrul Hasan Khan

"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(30)

five_year = 2019

data_all = np.load(f'US_census_{five_year}_median_home_value_data.npy')

## postions of missing values in the response
pos_nan = (np.asarray(np.where(np.isnan(data_all[:, 0])))).reshape(-1, )

## data where the response is missing
data_predict = data_all[pos_nan, :]

## data where all response is available
data = np.delete(data_all, pos_nan, 0)

## creating covariate matrix and response

X_numpy = data[:, 1:]
y_numpy = data[:, 0]
y_numpy = y_numpy.reshape(y_numpy.shape[0], 1)

## converting numpy to tensor

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

## standardizing the covariate matrix
means = X.mean(dim=1, keepdim=True)
stds = X.std(dim=1, keepdim=True)
X_standardized = (X - means) / stds

y_log = torch.log(y)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y_log, test_size=0.2, random_state=123456)

n_samples, n_features = X.shape

# Hyper-parameters 
input_size = n_features
hidden_1_size = 100 
hidden_2_size = 500 
hidden_3_size = 50 
output_size = 1
num_epochs = 1000
learning_rate = 0.01


# Fully connected neural network with three hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_1_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_1_size, hidden_2_size)  
        self.relu = nn.ReLU()
        #self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.l3 = nn.Linear(hidden_2_size, hidden_3_size)  
        self.relu = nn.ReLU()
        #self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.l4 = nn.Linear(hidden_3_size, output_size)  

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        # no activation and no MSELoss at the end
        return out

# 1) Model
model = NeuralNet(input_size, hidden_1_size,hidden_2_size, hidden_3_size, output_size)

# 2) Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    

# 3) Training loop

for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # zero grad before the step
    optimizer.zero_grad()
    
    # Backward pass and update
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        #print(f'epoch: {epoch+1}, loss = {loss.item()/10**10:.4f}')
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

predicted = model(X_standardized).detach().numpy()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    y_predicted = model(X_test)
    MSE = criterion(y_predicted.exp(), y_test.exp())
    abs_bias =(abs(y_predicted.exp() - y_test.exp())). mean()
    print(f'MSE: {MSE.item()/10**10:.4f}')
    print(f'Absolute Bias: {abs_bias.item():.4f}')

