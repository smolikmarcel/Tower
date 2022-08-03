#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mysql.connector
import numpy as np

import sklearn
import torch
import torch.nn as nn


# In[2]:



mydb = mysql.connector.connect(
  host="192.168.166.34",
  user="tower",
  password="tower",
  database="tower"
)


mycursor = mydb.cursor()

mycursor.execute("SELECT time_stamp FROM messages ORDER BY time_stamp DESC LIMIT 1;")

tim = (mycursor.fetchall())
print((tim[0][0]))

mycursor = mydb.cursor()

mycursor.execute("SELECT * from messages where time_stamp >= " + str(int(tim[0][0]) - 150) + " order by time_stamp")

data = mycursor.fetchall()

print(len(data))
#np.savetxt("tower1.csv", myresult, delimiter =", ", fmt ='% s')


# In[3]:


data_categories = []
#new_data.insert(0, ["time_stamp", "motor", "fan", "vibrations", "temperature", "temperature_pir"])
c = 1

motor = "NaN"
fan = "NaN"
vibrations = "NaN"
temperature = "NaN"
temperature_pir = "NaN"

for y in range(len(data) - 2):
    
    x = y + 1
    
    if  (data[x][2] == 1): motor = data[x][3]
    elif(data[x][2] == 2): fan = data[x][3]
    elif(data[x][2] == 3): vibrations = data[x][3]
    elif(data[x][2] == 4): temperature = data[x][3]
    elif(data[x][2] == 8): temperature_pir = data[x][3]
    
    if(data[x][1] != data[x + 1][1]): 
        
        data_categories.insert(c, [data[x][1], motor, fan, vibrations, temperature, temperature_pir])
        
        motor = "NaN"
        fan = "NaN"
        vibrations = "NaN"
        temperature = "NaN"
        temperature_pir = "NaN"
        c += 1

#np.savetxt("tower_prep.csv", new_data, delimiter =", ", fmt ='% s')


# In[4]:


count = 0
new_data = []
offset = 0
#new_data.insert(0, ("time_stamp", "motor", "fan", "vibrations", "temperature", "temperature_pir"))

for y in range(len(data_categories) - 2):
    x = y + 1
    diff = data_categories[x + 1][0] - data_categories[x][0] - offset
    
    if(diff > 100):
        
        offset += diff - 1
        diff = 1
    
    new_data.append([data_categories[x][0], data_categories[x][1], data_categories[x][2], data_categories[x][3], data_categories[x][4], data_categories[x][5]])
    
    for k in range(int(diff) - 1):
        new_data.append([data_categories[x][0] + k + 1, "NaN", "NaN", "NaN", "NaN", "NaN"])
        


# In[5]:


motor = "0"
fan = "0"


for x in range(len(new_data)):
    if(new_data[x][1] == "NaN"): new_data[x][1] = motor
    else: motor = new_data[x][1]

for x in range(len(new_data)):
    if(new_data[x][2] != "NaN"): fan = new_data[x][2]
    new_data[x][2] = str(fan)

    
    
    
    
"""
    
sum = 0
count = 0

for x in range(len(new_data)):
    if(new_data[x][3] != "NaN"): 
        count += 1
        #print(new_data[x][3])
        sum += float(new_data[x][3])

avg = sum/count


for x in range(len(new_data)): 
    if(new_data[x][3] == "NaN"): new_data[x][3] = float(avg)

        
"""

last_x = 0 
new_data[last_x][3] = 0
for x in range(len(new_data)):
    
    if(new_data[x][3] == "NaN"): last_x += 1
        
    else:
        if(last_x != 0):
            
            print(type(new_data[x][3]))
            print(new_data[x - last_x - 1][3])
            print(type(new_data[x - last_x - 1][3]))
            
            print(x)
            print(type(x))
            print(last_x)
            print(type(last_x))
            
            avg = (new_data[x][3] - new_data[x - last_x - 1][3]) / (last_x + 1)
            for k in range(last_x): new_data[x - last_x + k][3] = new_data[x - last_x - 1][3] + (k + 1)*avg
            last_x = 0
            
        
 
    
        
        
        
        
temp = 35.0
cc = 0

for x in range(len(new_data)):
    
    #x = y + 1
    #print(data[x][4])
    if(new_data[x][4] == "NaN"):
        
        cc += 1
    
    else:
        
        if(cc > 0):
            diff = (float(new_data[x][4]) - float(temp))/(cc + 1)
            #print(data[x][4])
            #print(cc)
                  
            for k in range(cc):
                new_data[x  - cc + k][4] = str(temp + (k + 1) * diff)
                #data[x  - 1][4]  = "99"
        cc = 0
        temp = float(new_data[x][4])



temp_pir = -100.0

for x in range(len(new_data)):
    
    if(new_data[x][5] == "NaN"): new_data[x][5] = str(temp_pir)
    else: temp_pir = float(new_data[x][5])

        
        
        
len(new_data)   
for y in range(len(new_data)):
    
    x = len(new_data) - 1 - y
    
    if(float(new_data[x][5]) < -99): new_data[x][5] = str(temp_pir)
    else: temp_pir = float(new_data[x][5])
        
        
        
        
        
        
time_predict = []
fan_stat = 0
time = 0    
for y in range(len(new_data)):
    
    x = len(new_data) - 1 - y
    
    if(float(new_data[x][5]) < -99): new_data[x][5] = str(temp_pir)
    else: temp_pir = float(new_data[x][5])

for x in range(len(new_data)):
    
    if(fan_stat == float(new_data[x][2])): time_predict.append(-100)
    else:
        fan_stat = float(new_data[x][2])
        time_predict.append(new_data[x][0])


        
    
    
    
    
time_predict[-1] = new_data[-1][0]

for y in range(len(time_predict)):
    
    x = len(time_predict) - 1 - y
    
    if(time_predict[x] != -100): time = time_predict[x]
    time_predict[x] = time - new_data[x][0]




data_merge = []
data_merge.append(("time_stamp", "motor", "fan", "vibrations", "temperature", "temperature_pir", "predicted_time"))

#print(len(new_data))
#print(len(time_predict))

for x in range(len(new_data)): 
    
    if(new_data[x][0] != "NaN" and new_data[x][1] != "NaN" and new_data[x][2] != "NaN" and new_data[x][3] != "NaN" and new_data[x][4] != "NaN" and new_data[x][5] != "NaN"):
        data_merge.append((new_data[x][0], new_data[x][1], new_data[x][2], new_data[x][3], new_data[x][4], new_data[x][5], time_predict[x]))
    
    #if(new_data[x][4] != "NaN"):
    #   data_merge.append((new_data[x][0], new_data[x][1], new_data[x][2], new_data[x][3], new_data[x][4], new_data[x][5], time_predict[x]))
    
    
    
np.savetxt("tower_prep_data.csv", data_merge, delimiter =", ", fmt ='% s')





