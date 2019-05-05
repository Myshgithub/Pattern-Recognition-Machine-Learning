#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


sess = tf.InteractiveSession()


# In[4]:


my_tensor = tf.random_uniform((4,4),0,1)


# In[6]:


my_tensor


# In[12]:


sess.run(my_tensor)


# Variables:::

# In[13]:


my_var = tf.Variable(initial_value=my_tensor)


# In[14]:


print(my_var)


# In[16]:


# sess.run(my_var)...............Note! You must initialize all global variables!


# In[21]:


# This line is really important, it is easy to forget!
init=tf.global_variables_initializer()


# In[26]:


sess.run(init)


# In[25]:


sess.run(my_var)


# In[27]:


my_var.eval()


# In[28]:


ph=tf.placeholder(tf.float32)


# In[29]:


# For shape its common to use (None,# of Features) 
# because None can be filled by number of samples in data
ph = tf.placeholder(tf.float32,shape=(None,5))


# In[31]:


print(ph)


# In[ ]:




