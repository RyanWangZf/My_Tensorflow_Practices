# -*- coding: utf-8 -*-

# 后续待实现功能:
# 1. Adaptive Learning rate
# 2. Mini-batch
# 3. Batch Normalization
# 4. Loss Function

import numpy as np
import pdb

class NeuralNetworks(object):
    
    def __init__(self):
        self.activation = lambda x: 1/(1+np.exp(-x))
        self.deriv_activation = lambda x: x*(1-x)
        
        
    def train_op(self,x,y):
            
        # forward pass
        l0 = x # shape = (n_samples,n_features)
        l1 = self.activation(np.dot(l0,self.weight0)) # (n_samples,hidden_units1) 
        l2 = self.activation(np.dot(l1,self.weight1)) # (n_samples,hidden_units2)
        l3 = self.activation(np.dot(l2,self.weight2)) # (n_samples,1)
        
        # backward pass
        l3_err = y - l3 # (n_samples,1)
        l3_delta = l3_err * self.deriv_activation(l3) # (n_samples,1)
        
        l2_err = l3_delta.dot(self.weight2.T) # (n_samples,hidden_units2)
        l2_delta = l2_err * self.deriv_activation(l2) # (n_samples,hidden_units2)
        
        l1_err = l2_delta.dot(self.weight1.T) # (n_samples,hidden_units1)
        l1_delta = l1_err * self.deriv_activation(l1) # (n_samples,hidden_units1)
        
        # update weights
        self.weight2 += np.dot(l2.T,l3_delta) * self.learning_rate # (hidden_units2,1)
        self.weight1 += np.dot(l1.T,l2_delta) * self.learning_rate  # (hidden_units1,hidden_units2)
        self.weight0 += np.dot(l0.T,l1_delta) * self.learning_rate # (n_features,hidden_units1)
        
        return l3_err
    
    def fit(self,x,y,n_epochs=10000,learning_rate=0.1):
        
        # set layer & num of hidden units
        hidden_units1 = 8
        hidden_units2 = 4
        self.weight0 = 2*np.random.random((x.shape[1],hidden_units1)) - 1 # shape = (n_features,hidden_units1)
        self.weight1 = 2*np.random.random((hidden_units1,hidden_units2)) - 1 # shape = (hidden_units1,hidden_units2)
        self.weight2 = 2*np.random.random((hidden_units2,1)) - 1 # shape = (hidden_units2,1)
        
        # initialize
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # training on epochs
        for i in range(n_epochs):
            err_ar = self.train_op(x,y)
            loss = (err_ar**2).sum()
            print("{} epoch's loss: {}".format(str(i),str(loss)))
    
    def predict(self,x):
        # forward pass
        l0 = x # shape = (n_samples,n_features)
        l1 = self.activation(np.dot(l0,self.weight0)) # (n_samples,hidden_units1) 
        l2 = self.activation(np.dot(l1,self.weight1)) # (n_samples,hidden_units2)
        predy = self.activation(np.dot(l2,self.weight2)) # (n_samples,1)
        return predy
        
if __name__ == "__main__":

    X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])
    
    y = np.array([[0],
                [1],
                [1],
                [0]])
    
    nn = NeuralNetworks()
    nn.fit(X,y,10000,1)
    print("true target:\n",y.flatten())
    print("fitted target:\n",nn.predict(X).flatten())
    pdb.set_trace()
        
        
        
        
        
        
        