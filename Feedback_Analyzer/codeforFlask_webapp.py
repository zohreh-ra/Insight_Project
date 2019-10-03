# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:58:23 2019

@author: Bahareh
"""
""" this code analyzes the feedback that has been inserted into the webapp (Flask)
and returns the results"""

def app_output(this_feedback):
    import numpy as np, pandas as pd
    import os
    import nltk
    import spacy
    import pickle
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize
    from nltk.corpus import words
    from spacy.lang.en.stop_words import STOP_WORDS
    from textblob import Word
    from sklearn.cluster import KMeans
    from nltk.tag import pos_tag
    from collections import Counter
    import matplotlib.pyplot as plt
    from nltk.corpus import wordnet 
    from sklearn.neighbors import KNeighborsClassifier
    import gensim
    import gensim.models
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    
    #dict_words = words.words()
    
    def is_english_word(word):
        dictionary = dict.fromkeys(dict_words, None)
        x = dictionary[word]
    
    def    first_clean(this_feedback):
        words=word_tokenize(str(this_feedback))
        words = [w for w in words if is_english_word(w)]  #takes forever to run
        return " ".join(w for w in words) 
    
    def avg_num_characters(this_feedback):
      words=word_tokenize(str(this_feedback))
      return (sum(len(word) for word in words)/len(words))
    
    def clean_text(this_feedback):
        words=word_tokenize(str(this_feedback))
        words = [w for w in words if w.isalpha()] # filters out ? ! , . 've 'r
        words = [w.lower() for w in words]
        words = [w for w in words if len(w) > 1] #filter out short tokens (this will delete I and punctuations as well)
        #words = [w for w in words if is_english_word(w)]  #takes forever to run
        return " ".join(w for w in words) 
    
    def part_speech(this_feedback):
        words=word_tokenize(str(this_feedback))
        #nltk.ne_chunk(words)
        return nltk.pos_tag(words)
    
    def  count_nouns(pos_count):   #micheal,..
        for a, b in pos_count.items():
            if a == 'NNP'or a == 'NNPS':
                return b
            else:
                return 0
            
    def  count_derterminer(pos_count): #his/her/that
        for a, b in pos_count.items():
            if a == 'DT':
                return b
            else:
                return 0
     
    def  count_wh_det(pos_count): #what/whose/when
        for a, b in pos_count.items():
            if a == 'WDT':
                return b
            else:
                return 0
     
    def  count_listM(pos_count): #listMaker 1)...
        for a, b in pos_count.items():
            if a == 'LS':
                return b
            else:
                return 0
            
    def remove_punctuation(sometext):
        '''a function for removing punctuation'''
        import string
        # replacing the punctuations with no space  
        replace_punct = str.maketrans('', '', string.punctuation)
        return sometext.translate(replace_punct)         
    
    
    
    #tokenizations
    words_seppunct = word_tokenize(str(this_feedback)) #punctuation seperated and space
    words_sepspace= this_feedback.split(" ") 
    
    #number of words
    word_count = len(words_sepspace)
    
    #number of characters
    char_count = len(this_feedback) ## 
    
    avg_num_characters = avg_num_characters(this_feedback)
    
    stopwordss = set(stopwords.words('english'))
    nlp = spacy.load('en_core_web_sm')
    stop_words=(nlp.Defaults.stop_words).union(stopwordss)
    #nlp.Defaults.stop_words.add(stopwordss)
    stopwords_count = len([x for x in word_tokenize(str(this_feedback)) if x in stop_words])
    #normalized number of stopwords
    normalized_stopwords_count = stopwords_count / word_count
    Norm_avg_num_characters= (avg_num_characters-1)/34
    #number of uppercase words
    uppercase = len([x for x in str(this_feedback).split() if x.isupper()])
    
    POS =part_speech(this_feedback)
    pos_count= Counter([j for i,j in POS])  
    ##feedback_data['total_pos_count'] = feedback_data['pos_count'].apply(lambda x:sum(x.values()))
    
    count_propernoun=count_nouns(pos_count)
    count_determiner=count_derterminer(pos_count)
    count_wh_det=count_wh_det(pos_count)
    cleaned_feedback = clean_text(this_feedback)
    
    ## load training data and train KNN model and save it for future prediction:------------------
    #------
    #features_train=pd.read_csv('Train_data_features.csv', encoding='utf-8')
    #labels_train=pd.read_csv('Train_data_labels.csv', encoding='utf-8')
    #feedback_data=pd.read_csv('Train_data3.csv', encoding='utf-8')
    #------
    filename='knn_train_model_saved.sav'
    #model_knn = KNeighborsClassifier(n_neighbors=5)
    #model_knn.fit(features_train,labels_train.values.ravel())
    #pickle.dump(model_knn, open(filename, 'wb')) #save the model
    ##---------------------------------------------------------------------
    
    ## train doc2vec model and save it , to use it for inferring this feedback's doc2vec:=----------
    #--------
    #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(feedback_data['Feedback_clean'].apply(lambda x: x.split(" ")))]
    ## train a Doc2Vec model text data and transform document into a vector data
    #model_dv = Doc2Vec(documents, vector_size=512, window=2, min_count=1, workers=4)
    #len(model_dv.docvecs) #12517
    #model_dv.save('doc2vec_saved') #save the model
    ##--------------------------------------------------------------------------
    
    feature_topredict=[Norm_avg_num_characters, normalized_stopwords_count ]
    model_dv = gensim.models.doc2vec.Doc2Vec.load('doc2vec_saved')  # load the model that is already there
    doc2vec_thisfeed = model_dv.infer_vector(cleaned_feedback.split(" "))
    # docvec = model_dv.docvecs[1] #printing the vector of document at index 1 
    # similar_doc = model_dv.docvecs.most_similar(14) #to get most similar document with similarity scores using document-index
    #doc2vec_thisfeed = model_dv.docvecs.infer_vector(cleaned_feedback.split(" ")) #get vector of document 
    feature_topredict=pd.concat([pd.DataFrame(feature_topredict),pd.DataFrame(doc2vec_thisfeed)],axis=0) #update features
    
    ##load the model from disk and Predict Output-------------------------------
    loaded_model_knn = pickle.load(open(filename, 'rb')) 
    predicted= loaded_model_knn.predict((feature_topredict.T)) 
    #print(predicted)
    ##---------------------------------------------------------------------------------
    if predicted ==0:
        quality_text= " Awesome! your feedback is specific"
    else:
        quality_text= " your feedback is vague! Please try to be more specific"

    return quality_text


