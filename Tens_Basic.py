#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


tf.VERSION


# In[4]:


print(tf.__version__)


# In[5]:


hello=tf.constant("Hello ")


# In[6]:


world=tf.constant("World")


# In[7]:


type(hello)


# In[8]:


print(hello)


# In[9]:


with tf.Session() as sess0:
    result= sess0.run(hello + world)


# In[10]:


print(result)


# In[12]:


a = tf.constant(10)
b = tf.constant(20)


# In[13]:


a+b


# In[14]:


a+b


# In[15]:


a-b


# In[17]:


with tf.Session() as sess:
    Res= sess.run(a+b)


# In[18]:


Res


# In[19]:


print(Res)


# In[20]:


const=tf.constant(10)


# In[38]:


fill_mat = tf.fill((3,3),11)  #3 by 3 fill with 11
fill_mat


# In[45]:


myzeros = tf.zeros((3,3))


# In[46]:


myones = tf.ones((3,3))


# In[47]:


myrandn = tf.random_normal((3,3), mean=0, stddev=1.0)


# In[48]:


myrandu = tf.random_uniform((3,3),minval=0, maxval=1)


# In[ ]:





# In[49]:


my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu] #list of all operations


# In[58]:


with tf.Session() as sess: #One way OR ...interactive session for Jupyter Notebook just...
    for op in my_ops:
     R1= sess.run(op)  #OR op.eva()
     print (R1)
     print('\n')


# In[56]:


print(R1)


# In[59]:


sess02 = tf.InteractiveSession() #interactive session for Jupyter Notebook jus


# In[60]:


for op in my_ops:
    print(sess02.run(op))
    print('\n')


# In[62]:


a =tf.constant([[1,2],   #Matrixes..., Multiply,...
               [3,4]])


# In[63]:


a.get_shape()


# In[64]:


b = tf.constant([[10],[100]])


# In[65]:


b.get_shape()


# In[67]:


c= tf.matmul(a,b)
sess02.run(c)  #Because of defining Interactive session! with no with


# In[68]:


c.eval()


# In[ ]:




