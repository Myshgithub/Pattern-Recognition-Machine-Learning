#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Regression Example

# ## Creating Data

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[12]:


# 1 Million Points
x_data = np.linspace(0.0,10.0,1000000)


# In[13]:


noise = np.random.randn(len(x_data))


# In[14]:


# y = mx + b + noise_levels
b = 5

y_true =  (0.5 * x_data ) + 5 + noise


# In[15]:


x_df=pd.DataFrame(data=x_data, columns=['X Data'])


# In[16]:


print(x_df)


# In[22]:


y_df=pd.DataFrame(data=y_true, columns=['Y'])


# In[26]:


y_df.head()


# In[28]:


my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)


# In[29]:


my_data.head()  # The first top 5 first


# In[46]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')


# # TensorFlow
# ## Batch Size
# 
# We will take the data in batches (1,000,000 points is a lot to pass in at once)

# In[47]:


import tensorflow as tf


# In[48]:


# Random 10 points to grab
batch_size = 8


# ** Variables **

# In[49]:


m = tf.Variable(0.5)
b = tf.Variable(1.0)


# ** Placeholders **

# In[50]:


xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])


# ** Graph **

# In[51]:


y_model = m*xph + b


# ** Loss Function **

# In[52]:


error = tf.reduce_sum(tf.square(yph-y_model))


# ** Optimizer **

# In[53]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


# ** Initialize Variables **

# In[54]:


init = tf.global_variables_initializer()


# ### Session

# In[55]:


with tf.Session() as sess:
    
    sess.run(init)
    
    batches = 1000
    
    for i in range(batches):
        
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        
        sess.run(train,feed_dict=feed)
        
    model_m,model_b = sess.run([m,b])


# In[56]:


model_m


# In[57]:


model_b


# ### Results

# In[68]:


y_hat = x_data * model_m + model_b


# In[69]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'C')


# ## tf.estimator API
# 
# Much simpler API for basic tasks like regression! We'll talk about more abstractions like TF-Slim later on.

# In[70]:


feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]


# In[71]:


estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


# ### Train Test Split
# 
# We haven't actually performed a train test split yet! So let's do that on our data now and perform a more realistic version of a Regression Task

# In[77]:


from sklearn.model_selection import train_test_split


# In[78]:


x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3, random_state = 101)


# In[79]:


print(x_train.shape)
print(y_train.shape)

print(x_eval.shape)
print(y_eval.shape)


# ### Set up Estimator Inputs

# In[80]:


# Can also do .pandas_input_fn
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)


# In[81]:


train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)


# In[82]:


eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False)


# ### Train the Estimator

# In[83]:


estimator.train(input_fn=input_func,steps=1000)


# ### Evaluation

# In[84]:


train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)


# In[85]:


eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)


# In[86]:


print("train metrics: {}".format(train_metrics))
print("eval metrics: {}".format(eval_metrics))


# ### Predictions

# In[87]:


brand_new_data=np.linspace(0,10,10)


# In[88]:


brand_new_data


# In[89]:


input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0,10,10)},shuffle=False)


# In[92]:


estimator.predict(input_fn=input_fn_predict)


# In[90]:


list(estimator.predict(input_fn=input_fn_predict))


# In[93]:


predictions = []# np.array([])
for x in estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])


# In[81]:


predictions


# In[95]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'r')


# # Great Job!
