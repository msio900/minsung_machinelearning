#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


pickle.load(open('./saves/favorite_save.pkl', 'rb'))


# In[3]:


favorite_load = pickle.load(open('./saves/favorite_save.pkl', 'rb'))
print(favorite_load)


# In[4]:


type(favorite_load)


# In[5]:


print(favorite_load['tiger'])


# In[7]:


autompg_lr = pickle.load(open('./saves/autompg_lr.pkl', 'rb')) # 오브젝트를 담을 때 쓰는 binary 라는 형식을 말함.
print(autompg_lr)


# In[8]:


type(autompg_lr)


# In[9]:

# input from outside
a = 3504.0
b = 8

import numpy as np

pre = np.array([[a,b]])
print(autompg_lr.predict(pre))


print(autompg_lr.predict([[3504.0,8]]))


# In[ ]:




