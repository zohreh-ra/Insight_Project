# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:41:51 2019

@author: Bahareh
"""

import numpy as np, pandas as pd
import os
import NLTK
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# load data
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
#os.getcwd()
feedback_data= pd.read_csv('FeedbacksSpecificityTone.csv', header=None, encoding='utf-8', names=["id", "Feedback"])
 
feedback_data.drop(feedback_data[(feedback_data['Feedback'] == ' ')  ].index , inplace=True)
feedback_count=feedback_data.shape[0]
feedback_data.head (10)

    
#function to remove punctuations
def remove_punctuation(sometext):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space  
    replace_punct = str.maketrans('', '', string.punctuation)
    return sometext.translate(replace_punct)


feedback_data['Feedback'] = feedback_data['Feedback'].apply(remove_punctuation)
#data.head(10)


# use 20% of data for testing
feedback_train_data, feedback_test_data = train_test_split(feedback_data,test_size=0.2,random_state=1234
print("Number of observations in Training Data ",len(feedback_train_data))
print("Number of observations in Testing Data ",len(feedback_test_data))




stop_words = set(stopwords.words('english'))



