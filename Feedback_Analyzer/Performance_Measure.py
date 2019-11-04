# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:41:51 2019

@author: Zohreh
"""
"""this code uses K-nearest Neighbors technique to evaluate
performance of final model against manually labelled data"""

import os
import pandas as pd
from sklearn import metrics
from collections import Counter
import pickle
import funcsigs
from funcsigs import signature
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve

##---load test data-----------
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
feedback_test= pd.read_csv('Test_data3.csv', encoding='utf-8')
#compute feature set for test data corresponding to features of train data
#feature_pos_test=pd.concat([feedback_test['Norm_NER_count'],feedback_test['Norm_count_propernoun'],feedback_test['Norm_count_wh_det'],feedback_test['Norm_count_determiner'],feedback_test['Norm_avg_num_characters'],feedback_test['normalized_stopwords_count'],feedback_test['Norm_count_poss_end'],feedback_test['Norm_count_num'],feedback_test['Norm_uppercase_count'],feedback_test['sentiments_compound']], axis=1)  
feature_pos_test=pd.concat([feedback_test['Norm_NER_count'],[feedback_test['Norm_count_determiner'],feedback_test['normalized_stopwords_count'],feedback_test['Norm_avg_num_characters']], axis=1)  
feature_pos_test=pd.concat([feature_pos_test,feedback_test[feedback_test.columns[30:542]]],axis=1)

#k-neighnourhood prediction ( based on previously saved and trained KNN model on training data)
filename='knn_train_model_saved.sav'
loaded_model_knn = pickle.load(open(filename, 'rb')) 
predicted= loaded_model_knn.predict(feature_pos_test) 
print(predicted)
Counter(predicted) 
# save test data and its features and predicted labels to file
feedback_test['predicted_label']=predicted
feedback_test.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\Test_data4.csv', index=False, encoding='utf-8')

##----------Confusion Matrix and F1 score calculation------------------
labelled= pd.read_csv('Test_data_tolabel_5.csv', encoding='utf-8')  # manually labelled data
predicted=feedback_test['predicted_label'] # 1,250 manually labelled data
actual=labelled['actual']

# I computed confusion matrix and precision and recall myself; but you can simply use sklearn.metrics to calculate them
true_pos=0
true_neg=0
false_pos=0
false_neg=0

for p,g in zip(predicted, actual): #0: vague feedback; 1: specific feedback
    if p==1 and g==1:
        true_pos+=1  #484
    if p==0 and g==0:
        true_neg+=1 #282
    if p==1 and g==0:
        false_pos+=1 #311
    if p==0 and g==1:
        false_neg+=1 #171
#655 : actual positives    ; #593  : actual negative
#795 : predicted positive  ; #453  : predicted negative   

precision_model= true_pos/(true_pos+false_pos)  #0.6088
recall_model=true_pos/(true_pos+false_neg)  #0.7389
f_1=2*(precision_model *recall_model)/ (precision_model+recall_model) #0.667586
print("Accuracy:",metrics.accuracy_score(actual, predicted))  #0.61378 (accuracy calculation)

##---------metrics to evaluate performance of clustering-------------
# zero is bad; 1 is good ;
homogeneity=metrics.homogeneity_score(actual, predicted)    
completeness= metrics.completeness_score(actual, predicted)  
v_measure=metrics.v_measure_score(actual, predicted)   
#perfect labelling==1, bad labelling closer to 0
ami= metrics.adjusted_mutual_info_score(actual, predicted) 
nmi=metrics.normalized_mutual_info_score(actual, predicted) 
mutual_i= metrics.mutual_info_score(actual, predicted)  #this can be either positive or negaive (negative is bad)
#btwn -1 and 1; negative values are bad (independent labelings), similar clusterings have a positive ARI, 1.0 is the perfect match score.
ari=metrics.adjusted_rand_score(actual, predicted) 

# ROC curve
fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label = 1)
roc_auc = auc(fpr, tpr)
plt.figure(1, figsize = (15, 10))
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curves')
plt.legend(loc="lower right")
plt.show()

# PR curve
average_precision = average_precision_score(actual, predicted)
precision, recall, _ = precision_recall_curve(actual, predicted)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure(1, figsize = (15, 10))
plt.step(recall, precision, alpha=0.2,where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve: Average Precision Score={0:0.2f}'.format(average_precision))