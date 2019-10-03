# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:41:51 2019

@author: Bahareh
"""

"""this code uses K-nearest Neighbors technique to compute
performance of final clustering model """


import numpy as np, pandas as pd
import os
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import pickle


##---load test data-----------
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
feedback_test= pd.read_csv('Test_data3.csv', encoding='utf-8')
#compute feature set for test data corresponding to features of train data
feature_pos_test=pd.concat([feedback_test['Norm_NER_count'],feedback_test['Norm_count_propernoun'],feedback_test['Norm_count_wh_det'],feedback_test['Norm_count_determiner'],feedback_test['Norm_avg_num_characters'],feedback_test['normalized_stopwords_count'],feedback_test['Norm_count_poss_end'],feedback_test['Norm_count_num'],feedback_test['Norm_uppercase_count'],feedback_test['sentiments_compound']], axis=1)  
feature_pos_test=pd.concat([feedback_test[feedback_test.columns[30:542]],feature_pos_test],axis=1)


#k-neighnourhood prediction ( based on previously saved and trained KNN model on training data)
filename='knn_train_model_saved.sav'
loaded_model_knn = pickle.load(open(filename, 'rb')) 
predicted= loaded_model_knn.predict(feature_pos_test) 
print(predicted)
Counter(predicted) 

# save test data and its features and predicted labels to file
feedback_test['predicted_label']=predicted
feedback_test.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\Test_data4.csv', index=False, encoding='utf-8')
#save test file to label manually and compare to prediction performance of the model
test_tolabel=pd.concat([feedback_test['Feedback'],feedback_test['predicted_label']], axis=1)
test_tolabel.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\Test_data_tolabel.csv', index=False, encoding='utf-8')


##----------Accuracy and F1 score calculation------------------
labelled= pd.read_csv('Test_data_tolabel2.csv', encoding='utf-8')  # after manually labelling the data
predicted=labelled['predicted']
actual=labelled['actual']

true_pos=0
true_neg=0
false_pos=0
false_neg=0

for p,g in zip(predicted, actual):
    if p==1 and g==1:
        true_pos+=1
    if p==0 and g==0:
        true_neg+=1
    if p==1 and g==0:
        false_pos+=1
    if p==0 and g==1:
        false_neg+=1

precision= true_pos/(true_pos+false_pos) 
recall=true_pos/(true_pos+false_neg)
f_1=2*(precision *recall)/ (precision+recall) 

#accuracy calculation
print("Accuracy:",metrics.accuracy_score(actual, predicted))
