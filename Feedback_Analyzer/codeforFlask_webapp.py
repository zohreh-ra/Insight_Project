# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:58:23 2019
web-app code:
This function takes the feedback as input and will predict its quality ( "vague" vs. "specific") 
and tone ( positive/ neutral/ negative) as output

@author: Bahareh
"""
def app_output(this_feedback):
    import pandas as pd
    import nltk
    import spacy
    import gensim
    import gensim.models
    import pickle
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize
    from spacy.lang.en.stop_words import STOP_WORDS
    from collections import Counter
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from sklearn.neighbors import KNeighborsClassifier
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from functions_preprocess import spell_check 
    from functions_preprocess import avg_num_characters
    from functions_preprocess import is_english_word
    from functions_preprocess import remove_nonEnglish
    from functions_preprocess import clean_text
    from functions_preprocess import ner_feedback
    from functions_preprocess import remove_punctuation
    from functions_preprocess import remove_stopwords
    from functions_preprocess import part_speech
    from functions_preprocess import count_nouns
    from functions_preprocess import count_derterminer
    from functions_preprocess import count_wh_det    
    from functions_visualization import sentiment_plot 
    	
    stopwordss = set(stopwords.words('english'))
    nlp = spacy.load('en_core_web_sm')
    stop_words=(nlp.Defaults.stop_words).union(stopwordss)
    
    # removing non-english words and correct spelling of words
    this_feedback = spell_check(this_feedback) 
    #feedback_data=remove_nonEnglish(this_feedback)  
    	
    #tokenization	
    words_seppunct = word_tokenize(str(this_feedback)) #punctuation seperated and space
    words_sepspace= this_feedback.split(" ")  
    #number of words
    word_count = len(words_sepspace) 
    #number of characters
    char_count = len(this_feedback)
    avg_num_characters = avg_num_characters(this_feedback)
    Norm_avg_num_characters= (avg_num_characters-1)/34 #normalized word characters count
    #number of uppercase words
    uppercase = len([x for x in str(this_feedback).split() if x.isupper()])
    #stopwords count
    stopwords_count = len([x for x in word_tokenize(str(this_feedback)) if x in stop_words])	
    normalized_stopwords_count = stopwords_count / word_count  #normalized number of stopwords
    #part-of-speech
    POS =part_speech(this_feedback)
    pos_count= Counter([j for i,j in POS])  
    count_propernoun=count_nouns(pos_count) #propernoun count
    count_determiner=count_derterminer(pos_count) # determiner count
    count_wh_det=count_wh_det(pos_count) #wh_determiner count   
    normalized_count_propernoun=count_propernoun / word_count #normalized number of propernouns
    normalized_count_determiner=count_determiner / word_count #normalized number of determiners
    normalized_count_wh_det=count_wh_det / word_count #normalized number of wh_determiners
    #Name Entity Recognition
    ner_count_feedback=0
    count_ner_labels=Counter(ner_feedback(this_feedback))
    for key in count_ner_labels:
        ner_count_feedback=ner_count_feedback+(count_ner_labels[key])  #total number of NER 
    #clean text
    cleaned_feedback = clean_text(this_feedback) 
    	
	##---------------------------------------------------------------------
	##---------------Preparing feature set -------------------------------------------------------
    feature_topredict=[ner_count_feedback,normalized_count_determiner,normalized_stopwords_count,Norm_avg_num_characters]
    model_dv = gensim.models.doc2vec.Doc2Vec.load('doc2vec_saved')  # load the previously saved doc2vec model, that is already there
    doc2vec_thisfeed = model_dv.infer_vector(cleaned_feedback.split(" ")) # infer vector of document for this feedback from saved doc2vec model
    # docvec = model_dv.docvecs[1] #the vector of document at index 1 
    # similar_doc = model_dv.docvecs.most_similar(14) #to get most similar document with similarity scores using document-index 
    feature_topredict=pd.concat([pd.DataFrame(feature_topredict),pd.DataFrame(doc2vec_thisfeed)],axis=0) #update features
    
    ##---------------load the previuosly saved KNN model from disk and Predict Output-------------
    filename='knn_train_model_saved.sav'
    loaded_model_knn = pickle.load(open(filename, 'rb')) 
    predicted= loaded_model_knn.predict((feature_topredict.T)) 
    #print(predicted)
    
    # start = '\033[1m{}\033[0m'
    #end = '\033[0m'
    if predicted == 0:		
    	quality_text= " your feedback is vague!Please try to be more specific."  
    	#quality_text= " Awesome! your feedback is" + start + "specific"+ end +"." 		
    elif predicted == 1:		
    	quality_text= " Awesome! your feedback is specific." 
    	#quality_text= " your feedback is" + start+ "vague!"+ end+ "Please try to be more specific."
    	   
	#------------display sentiment analysis output and its plot-----------   
    sid = SentimentIntensityAnalyzer()
    thisfeed_sentiments = sid.polarity_scores(this_feedback)
    names=[]
    values=[]
    for count, (key, value) in enumerate(thisfeed_sentiments.items(), 1):
    		
    	if key == 'neg':
    			neg_sent=value
    			names.append('negative')
    			values.append(value)
    	if key == 'neu':
    			neu_sent=value
    			names.append('neutral')
    			values.append(value)
    	if key == 'pos':
    			pos_sent=value
    			names.append('positive')
    			values.append(value)
    	 
    #horizontal sentiment plot with sentiment scores
    sent_plot=sentiment_plot(values, names)
    
    return quality_text,sent_plot