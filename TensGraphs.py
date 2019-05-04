#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


n1=tf.constant(1)


# In[3]:


n2=tf.constant(2)


# In[4]:


n3=n1+n2


# In[31]:


with tf.Session() as sess:
    #Res = sess.run(n3)  #n3.eval()
    #Or
    n3.eval()


# In[32]:


print (Res)


# In[33]:


print (n3)


# In[34]:


print(tf.get_default_graph())  #similarly we have default session as well


# In[35]:


g= tf.Graph()


# In[13]:


print(g)  #showing another Graph that has been made here!


# In[14]:


graph_one = tf.get_default_graph()
print (graph_one)


# In[17]:


graph_two = tf.Graph()
print (graph_two)


# In[18]:


#Making Graph Two instead here as Default one...


# In[20]:


with graph_two.as_default():
    print(graph_two is tf.get_default_graph())


# In[21]:


print(graph_two is tf.get_default_graph())


# In[ ]:




