# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:01:02 2016

@author: rob
"""
import numpy as np
from sklearn.metrics import confusion_matrix
acc_weights = np.array([0.964, 0, 0.958, 0, 0.964, 0])
test_weights = np.array([0.964, 0.958, 0.964])

#Load the assemble dataset
data_ass = np.genfromtxt('assembling_data.csv',delimiter = ',',skip_header=1)
y_ass = data_ass[:,1]

"""Save all the individual logits in a dictionary"""
# Note that every load-statement has different settings with respect to starting
# columns and rows. Adapt this to your need
logits = {}
logits_test = {}
logits['logits_1'] = np.genfromtxt('logits_nn_1hidden6apr.csv')
logits['logits_2'] = np.genfromtxt('Probabilities_Classes_LassoR1.csv',delimiter = ',',skip_header=1)[:,1:]
logits['logits_3'] = np.genfromtxt('predicted_probabilities_random_forest.csv',delimiter = ',',skip_header=1)[:,1:]
logits['logits_4'] = np.genfromtxt('predicted_probabilities_boosting.csv',delimiter = ',',skip_header=1)[:,1:]
logits['logits_5'] = np.genfromtxt('predicted_probabilities_bagging.csv',delimiter = ',',skip_header=1)[:,1:]
logits['logits_6'] = np.genfromtxt('Probabilities_Classes_RidgeR.csv',delimiter = ',',skip_header=1)[:,1:]
logits_test['logits_1'] = np.genfromtxt('logits_nn_1hidden_test8april.csv')
logits_test['logits_2'] = np.genfromtxt('random_forest_prediction_test2.csv',delimiter = ',',skip_header=1)[:,1:]
logits_test['logits_3'] = np.genfromtxt('bagging_probabilities_modelingTest.csv',delimiter = ',',skip_header=1)[:,1:]
Ntest = logits_test['logits_1'].shape[0]

#Expected sizes
D = 7
N = 3000
#Check these expected sizes
assert logits['logits_1'].shape == (N,D), 'Wrong size of logits_1'
assert logits['logits_2'].shape == (N,D), 'Wrong size of logits_2'
assert logits['logits_3'].shape == (N,D), 'Wrong size of logits_3'
assert logits['logits_4'].shape == (N,D), 'Wrong size of logits_4'
assert logits['logits_5'].shape == (N,D), 'Wrong size of logits_5'

#Perform weighted sum over individual logits
logits_weighted_sum = np.zeros((N,D))
for n in xrange(len(acc_weights)):
    logits_weighted_sum += acc_weights[n]*logits['logits_'+str(n+1)]
logits_weighted_sum /= np.sum(acc_weights)

#Perform weighted sum over individual logits over testset
logits_test_sum = np.zeros((Ntest,D))
for n in xrange(len(test_weights)):
    logits_test_sum += test_weights[n]*logits_test['logits_'+str(n+1)]
logits_test_sum /= np.sum(test_weights)

#Make predictions
pred = {}
acc = {}
conf = {}
ytrue = np.expand_dims(y_ass,axis=1)
for n in xrange(len(acc_weights)):
    logits_n = logits['logits_'+str(n+1)]
    pp = np.argmax(logits_n,axis=1)
    pred['classifier_'+str(n+1)] = pp
    ypp = np.expand_dims(pp,axis=1)
    print('Confusion matrix for classifier %s'%(n+1))
    print(confusion_matrix(ytrue,ypp))
    #Save the accuracy for later printing
    acc['classifier_'+str(n+1)] = np.mean(ytrue==ypp)
    #Calculate the average confidence at the falsely classified samples
    ind_false = np.where(ytrue!=ypp)
    ind_false = ind_false[0]
    class_false = np.squeeze(ytrue[ind_false]).astype(int)
    conf_false = logits_n[ind_false,class_false]
    conf['classifier_'+str(n+1)] = np.mean(conf_false)

#Print the accuracies
for n in xrange(len(acc_weights)):
    print('Accuracy for classifier %s is %.3f'%(n+1,acc['classifier_'+str(n+1)]))
    
#Print the confidences
for n in xrange(len(acc_weights)):
    print('Average confidence at misclassified samples for classifier %s is %.3f'%(n+1,conf['classifier_'+str(n+1)]))
    

#Check if the weighted sum makes sense
assert np.linalg.norm(np.sum(logits_weighted_sum,axis=1)-1) < 0.001,'The weighted sum seems not to result in a probability distribution'

ensemble_pred = np.argmax(logits_weighted_sum,axis=1)
ensemble_pred = np.expand_dims(ensemble_pred,axis=1)
acc_ens = np.mean(ensemble_pred == ytrue)
assert len(ensemble_pred) == N, 'Something in the sizes of argmax faulted'
print('Ensemble accuracy is %.3f'%(acc_ens))


#Make predictions on the testset
test_pred = np.argmax(logits_test_sum,axis=1)
test_pred = np.expand_dims(test_pred,axis=1)
test_pred = np.concatenate((np.expand_dims(np.arange(1,20001,1),axis=1),test_pred),axis=1)

# Check the consistency of the different classifiers in the ensemble
pred_ens = {}
for n in xrange(len(test_weights)):
    logits_n = logits_test['logits_'+str(n+1)]
    pp = np.argmax(logits_n,axis=1)
    pred_ens['classifier_'+str(n+1)] = pp

consis1 = (pred_ens['classifier_1'] == pred_ens['classifier_2'])
consis2 = (pred_ens['classifier_1'] == pred_ens['classifier_3'])
consis3 = (pred_ens['classifier_2'] == pred_ens['classifier_3'])
consis = consis1 & consis2
consis = np.mean(consis)
print('\n')
print('Consistency 1 & 2 %.3f'%(np.mean(consis1)))
print('Consistency 1 & 2 %.3f'%(np.mean(consis2)))
print('Consistency 3 & 2 %.3f'%(np.mean(consis3)))
print('The three classifiers are consistent on %.3f'%(consis))


# Save the predictions for the testset
np.savetxt('prediction_24.csv',test_pred)