#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


fut_pred = 900
n_timesteps = 60


# In[3]:

dirname = os.path.dirname(os.path.abspath(__file__))
data = np.loadtxt(f'{dirname}/tower_prep_data_json.csv', delimiter=',', dtype=np.float32, skiprows=1)


# In[4]:


scaler_vibrations = joblib.load(f'{dirname}/../scalers/motor2.gz')
scaler_temperature = joblib.load(f'{dirname}/../scalers/temperature.gz')
scaler_temperature_pir = joblib.load(f'{dirname}/../scalers/temperature_pir.gz')
scaler_start_stop = joblib.load(f'{dirname}/../scalers/start_stop.gz')


# In[5]:


class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 30 # number of hidden states
        self.n_layers = 4 # number of LSTM layers (stacked)
    
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


# In[6]:


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


# In[7]:


motor_model = torch.load(f"{dirname}/../models/motor2.pt")
temperature_model = torch.load(f"{dirname}/../models/temperature2.pt")


# In[8]:


motor_model.eval()
temperature_model.eval()


# In[9]:


def motor_eval(vib):
    motor_model.eval()

    with torch.no_grad():

        outputs = torch.tensor(vib.reshape(-1, 60))

        outputs = motor_model(outputs)
        _, predicted = torch.max(outputs, 1)
        
        return predicted.item()


# In[10]:


size = 60
shift = 0

vibrations =  np.array( data[-size + shift:, 3] )
temperature =  np.array( data[-size + shift:, 4] )
temperature_pir =  np.array( data[-size + shift:, 5])

#vibrations = scaler_vibrations.transform(vibrations.reshape(-1, 1))
#temperature = scaler_temperature.transform(temperature.reshape(-1, 1))

vibrations = scaler_vibrations.transform(np.log( vibrations + 1e-15).reshape(-1, 1))
temperature = scaler_temperature.transform(temperature .reshape(-1, 1))
#temperature_pir = scaler_temperature_pir.transform(temperature_pir.reshape(-1, 1))
temperature_pir = temperature_pir/50


pred_vibrations = np.empty(temperature.size)
pred_vibrations.fill(motor_eval(vibrations)/5)
    
dataset_test = np.stack((pred_vibrations, temperature[:, 0], temperature_pir), axis=1)
#dataset = temperature[:, 0]
#dataset_test = np.stack((pred_vibrations, temperature, temperature_pir), axis=1)
#dataset = temperature[:, 0]


# In[11]:


for i in range(fut_pred):
    
    seq = torch.FloatTensor([dataset_test[i*3: i*3 + 3*n_timesteps].reshape(-1, 3)])
    #print(seq)
    #print(seq)

    with torch.no_grad():

        temperature_model.init_hidden(seq.size(0))
        dataset_test = np.append(dataset_test, [pred_vibrations[0], temperature_model(seq).item(), temperature_pir[-1]])
        #dataset_test = np.append(dataset_test, [pred_vibrations[0], temperature_model(seq).item(), temperature_pir[-1]] )


# In[12]:


#dataset_test = torch.FloatTensor(dataset_test)
#dataset_test_l = torch.FloatTensor(dataset_test_l)
#temp = scaler_temperature.inverse_transform(dataset_test.reshape(-1, 3)[-fut_pred:, 1].reshape(-1, 1))
    

    
#plt.grid(True)
#plt.autoscale(axis='x', tight=True)
#plt.plot(temp)
#plt.plot(dataset_test[:fut_pred, 1])
#plt.plot(temperature[:fut_pred])
#plt.show()


# In[ ]:



stop = timer()
print(stop - start)

