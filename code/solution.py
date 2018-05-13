import numpy as np
from helper import *
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
   
   feature_count = data.shape[1];
   n,_ = data.shape
   w = np.zeros(feature_count)
   
   for iterations in range(max_iter):
       gradient = 0
       for i in range(n):
           W = np.transpose(w)
           gradient = gradient + ((data[i]*label[i])/(1 + np.exp(label[i]*(W)*data[i])))
       gradient = gradient * (-1 / n)
       vector = gradient * (-1)
       
       w = w + (learning_rate * vector)
    
   return w
           
   '''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	'''
  


def thirdorder(data):
    n,_ = data.shape
    order = [[0 for x in range(10)] for y in range(n)]
    for i in range(n):  #order = (1, x1, x2, x2^2, x1x2, x2^2, x1^3, x1^2*x2, x1x2^2, x2^3)
        order[i][0] = 1
        order[i][1] = (data[i][0])
        order[i][2] = (data[i][1])
        order[i][3] = (data[i][0]**2)
        order[i][4] = (data[i][0]) * (data[i][1])
        order[i][5] = (data[i][1]**2)
        order[i][6] = (data[i][0]**3)
        order[i][7] = (data[i][0]**2) * (data[i][1])
        order[i][8] = (data[i][0]) * (data[i][1]**2)
        order[i][9] = (data[i][1]**3)
    return np.array(order)
    '''	This function is used for a 3rd order polynomial transform of the data.	
    Args:	data: input data with shape (:, 3) the first dimension represents 		  
    total samples (training: 1561; testing: 424) and the 		  
    second dimesion represents total features.	
    
    Return:		
        result: A numpy array format new data with shape (:,10), which using 		
        a 3rd order polynomial transformation to extend the feature numbers 		
        from 3 to 10. 		
    
    The first dimension represents total samples (training: 1561; testing: 424) 		
    and the second dimesion represents total features.	
    '''
    


def accuracy(x, y, w):
    n = len(y)
    sigmoid = 0
    threshold = 0.5
    correct = 0
    for i in range(n):
        W = np.transpose(w)
        sigmoid = 1 / (1 + np.exp(-1 * y[i] * np.dot(W, x[i])))
        if(sigmoid > threshold): classify = 1
        else : classify = -1
        
        if (classify == y[i]): correct = correct + 1
    
    accuracy = correct / n
    
    return accuracy
    
    
    
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    


