#!/usr/bin/env python
# coding: utf-8

# In[75]:


import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim

# multivariate data preparation
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt

import joblib 

import requests
import os
from timeit import default_timer as timer
start = timer()

import pickle

dirname = os.path.dirname(os.path.abspath(__file__))


data = np.loadtxt(f'{dirname}/tower_prep_data_json.csv', delimiter=',', dtype=np.float32, skiprows=1)

# In[77]:
print()
scaler_vibrations = joblib.load(f'{dirname}/../scalers/motor2.gz')
scaler_temperature = joblib.load(f'{dirname}/../scalers/temperature.gz')
scaler_temperature_pir = joblib.load(f'{dirname}/../scalers/temperature_pir.gz')
scaler_start_stop = joblib.load(f'{dirname}/../scalers/start_stop.gz')


# In[91]:
size = 60

test_data_x =  np.log( data[-size :, 3] + 1e-15)
test_data_y  =  data[-size :, 1]


# In[92]:
class NeuralNetwork(nn.Module):
    
    def __init__(self):
        
        super(NeuralNetwork, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
        
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 6)
        )
        
    
    def forward(self, x):
        
        output = self.linear_relu_stack(x)
        return output


# In[93]:
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20#30 # number of hidden states
        self.n_layers = 4#3 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)


# In[94]:
start_stop_model = torch.load(f"{dirname}/../models/start_stop2.pt")
motor_model = torch.load(f"{dirname}/../models/motor2.pt")


# In[95]:
motor_model.eval()

with torch.no_grad():
    
    outputs = torch.tensor(scaler_vibrations.transform(test_data_x.reshape(-1,1)).reshape(-1, 60))
    #label = int(test_data_y[size])
        

    outputs = motor_model(outputs)
    _, predicted = torch.max(outputs, 1)
        
    print(f"predicted motor status: {predicted[0].item()}")
    print(f"real motor status: {test_data_y[size-1]}")
    




# In[98]:
#vibrations =  data[-size:, 3]
temperature =  data[-size:, 4]
temperature_pir =  data[-size:, 5]
start_stop =  data[-size:, 6]

#print(temperature)

# In[99]:
#vibrations = scaler_vibrations.transform(vibrations.reshape(-1, 1))
temperature = scaler_temperature.transform(temperature.reshape(-1, 1))
temperature_pir = scaler_temperature_pir.transform(temperature_pir.reshape(-1, 1))


#temperature = scaler_temperature.transform(temperature.reshape(-1, 1))
#start_stop = scaler_start_stop.transform(dataset[:, 2] .reshape(-1, 1))
#print(temperature_pir)
temperature_pir = temperature_pir/50

# In[100]:
dataset_test = []

dataset_test = np.stack((temperature[-60:, 0], temperature_pir[-60:, 0]), axis=1)

#for x in range(len(temperature)):
#    dataset_test.append([ temperature[x][0], temperature_pir[x][0] ])


# In[101]:
start_stop_model.eval()
seq = torch.FloatTensor([ dataset_test ])


with torch.no_grad():

    start_stop_model.init_hidden(seq.size(0))
    temp = scaler_start_stop.inverse_transform(np.array(start_stop_model(seq).item()).reshape(-1, 1))

    print(f"predicted time to stop: {temp}")
    print(f"real time to stop: {start_stop[-1]}")



requests.post('http://192.168.166.39:1880/motor_status', data = { "payload": str(predicted[0].item()), "topic":"motor_status" })
requests.post('http://192.168.166.39:1880/time_to_switch', data = { "payload": str(temp[0].item()), "topic":"time_to_switch" })

stop = timer()
print(f"time: {stop - start}")


#f.close()
    
    
