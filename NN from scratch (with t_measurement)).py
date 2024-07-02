#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:28:31 2024

@author: colomb
"""
# %% 1 

#package imports
import matplotlib.pyplot as plt 
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
from IPython import get_ipython
import time
from contextlib import contextmanager

# function to measure the execution time of a block
@contextmanager
def timed_block(name):
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for {name}: {execution_time:.4f} seconds")

# Display plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

# use timed_block to record the entire execution time
with timed_block("Entire Script"):   
    # %% 2
    # Generate a dataset and plot it
    np.random.seed(7)
    X, y = sklearn.datasets.make_moons(800, noise=0.30)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title('Dataset with increase noise')
    plt.show()
    
    # %%3
    # train logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV() 
    clf.fit(X, y) 
    
    # #helper function to plot a decision boundary
    def plot_decision_boundary(pred_func): 
         # Set min and max values and give it some padding 
         x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
         y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
         h = 0.01 
         # Generate a grid of points with distance h between them 
         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
         # Predict the function value for the whole gid 
         Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
         Z = Z.reshape(xx.shape) 
         # Plot the contour and training examples 
         plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
         plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 
    
    # plot a decision boundary
    plot_decision_boundary(lambda x: clf.predict(x))
    plt.title("logistic regression")
        
    # %%4 
    # implementation
    num_examples = len(X) #training set size
    nn_input_dim = 2 #input layer dimensionality
    nn_output_dim = 2 #output layer dimensionality
    
    # gradrient descent parameters 
    epsilon = 0.01 #learning rate for GD
    reg_lambda = 0.01 #regularization strength
    
    # helper function to evaluate the total loss on the dataset
    def calculate_loss (model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        #forward propagat to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        #add regularization term to loss
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./num_examples * data_loss
    
    # helper function to predict a label (0 or 1)
    def predict (model, x):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        #forward prop
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis = 1)
    
    # this function learns parmeters for the neural network and returns the model
    # - nn_dhim = number of nodes in the hl
    # - num_passes = number of passes through the data for the gradient descent
    # - print_loss = if True, print the loss every 1000 iterations
    def build_model (nn_hdim, num_passes = 20000, print_loss = False):
        
        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))
        
        # what we return at the end
        model = {}
        
        # gradient descent. for each batch:
        for i in range(0, num_passes):
            
           #forward propagation 
           z1 = X.dot(W1) + b1
           a1 = np.tanh(z1)
           z2 = a1.dot(W2) + b2
           exp_scores = np.exp(z2)
           probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
           
           # backpropagation
           delta3 = probs
           delta3[range(num_examples), y] -= 1
           dW2 = (a1.T).dot(delta3)
           db2 = np.sum(delta3, axis=0, keepdims=True)
           delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
           dW1 = np.dot(X.T, delta2)
           db1 = np.sum(delta2, axis=0)
        
           # Add regularization terms (b1 and b2 don't have regularization terms)
           dW2 += reg_lambda * W2
           dW1 += reg_lambda * W1
        
           W1 += -epsilon * dW1
           b1 += -epsilon * db1
           W2 += -epsilon * dW2
           b2 += -epsilon * db2
        
           # Assign new parameters to the model
           model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
           
           # to print the loss
           # it is expensive b/c uses the whole dataset, so we only do it every 1000 iterations
           if print_loss and i % 1000 == 0:
              print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))
              
        return model
        
    
    # %%5
    # build a model with a 3-d hiddel layer
    model = build_model(3, num_passes=21000, print_loss=True)
    
    
    # #plot the decision boundary
    # plot_decision_boundary(lambda x: predict(model, x))
    # plt.title("decision boundary for hidden layer size 3")
    
    #plot the decision boundary with the modified epsilon
    plot_decision_boundary(lambda x: predict(model, x))
    plt.title("Decision boundary (hl=3, epochs=20k)")
    
    # %%6
    #this loop varies the hidden layer dimensions and plot the effect on the decision boundary
    '''plt.figure(figsize=(16, 32))
    hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer size %d' % nn_hdim)
        model = build_model(nn_hdim)
        plot_decision_boundary(lambda x: predict(model, x))
    plt.show()'''
        
        
        
        
        
        
        
        
        
        
        