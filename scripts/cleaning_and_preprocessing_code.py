# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:41:51 2019

This code was used for cleaning and pre-processing of Feedback text documents
@author: Zohreh
"""
import os
import nltk
import spacy
import gensim
import string
import numpy as np, pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import words as nltk_words
from nltk.corpus import wordnet 
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tag import pos_tag
from nltk import FreqDist
from collections import Counter
from spacy import displacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from functions_visualization import show_wordcloud
from functions_visualization import plot_hist
from functions_visualization import plot_sentimentscore_frequency
from functions_visualization import plot_word_frequency
from functions_visualization import plot_nonlog_word_frequency
from functions_visualization import draw_sentiment_barplot

##apppend stopwords from different libraries (slightly different in each library; to have a more comprehensive coverage)
stopwordss = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
stop_words=(nlp.Defaults.stop_words).union(stopwordss)

##English vocabulary;
#myDict=wordnet.words()
#dict_words = nltk_words.words()
dictionary = dict.fromkeys(nltk_words.words(), None) 
spell = SpellChecker()

# load data
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
feedback_data_all= pd.read_csv('FeedbacksSpecificityTone.csv', encoding='utf-8',header=None, names=["id", "Feedback"])
feedback_data_all.drop_duplicates(subset ="Feedback", keep = 'first', inplace = True)
feedback_data_all=feedback_data_all.reset_index(drop=True)
 
# removing non-english words and blank rows and correct spelling of words
feedback_data_all['Feedback'] = feedback_data_all['Feedback'].apply(lambda x: spell_check(x))  
#feedback_data_all['Feedback'] = feedback_data_all['Feedback'].apply(lambda x: remove_nonEnglish(x))  
feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback'] == ' ') | (feedback_data_all['Feedback'] == '') ].index , inplace=True)
feedback_data_all=feedback_data_all.reset_index(drop=True)
#tokenized words for each feedback
feedback_data_all['words_seppunct'] = feedback_data_all['Feedback'].apply(lambda x: word_tokenize(str(x))) #punctuation seperated and space
feedback_data_all['words_sepspace'] = feedback_data_all['Feedback'].apply(lambda x: str(x).split(" "))  # only space seperated
 #POS extraction
feedback_data_all['POS']=np.zeros(len(feedback_data_all))       
feedback_data_all['POS'] =feedback_data_all['Feedback']. apply(lambda x: part_speech(x))     
#NER Extraction
ner_count_feedback=np.zeros(len(feedback_data_all))
for i in range(len(feedback_data_all)):
    labels_en = ner_feedback(feedback_data_all['Feedback'][i])
    count_ner_laels=Counter(labels_en)
    for key in count_ner_laels:
        ner_count_feedback[i]=ner_count_feedback[i]+(count_ner_laels[key])  #total number of NER 
feedback_data_all['ner_count_feedback'] =ner_count_feedback  
#number of stopwords
feedback_data_all['stopwords_count']  = feedback_data_all['Feedback'].apply(lambda x: len([x for x in word_tokenize(str(x)) if x in stop_words])) #min=0; max=204
#number of uppercase words
feedback_data_all['uppercase_count'] = feedback_data_all['Feedback'].apply(lambda x: len([x for x in str(x).split() if x.isupper()])) #min=0; max=17
#number of characters
feedback_data_all['char_count'] = feedback_data_all['Feedback'].str.len() ## this also includes spaces
#first step text cleaning
feedback_data_all['Feedback_clean']= feedback_data_all['Feedback'] .apply(lambda x: first_clean(x))
feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback_clean'] == ' ') | (feedback_data_all['Feedback_clean'] == '')].index , inplace=True)
feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback'] == ' ') | (feedback_data_all['Feedback'] == '')].index , inplace=True)
feedback_data_all=feedback_data_all.dropna(subset=['Feedback'])
feedback_data_all=feedback_data_all.dropna(subset=['Feedback_clean'])
feedback_data_all=feedback_data_all.reset_index(drop=True)

##--------------------extracting some features-----------------------------
words_clean=feedback_data_all['Feedback_clean'].apply(lambda x: word_tokenize(str(x)))
feedback_data_all['word_count']=words_clean.apply(lambda x: len(x)) #max=342
#avg number of characters in the words of a feedback= average word length
feedback_data_all['avg_num_characters'] = feedback_data_all['Feedback_clean'].apply(lambda x: avg_num_characters(x)) #min=1; max=34
feedback_data_all['Norm_avg_num_characters']= (feedback_data_all['avg_num_characters']-1)/34
#stopword count
feedback_data_all['normalized_stopwords_count'] = feedback_data_all['stopwords_count'] / feedback_data_all['word_count']
feedback_data_all['Norm_uppercase_count'] = feedback_data_all['uppercase_count']/17

##Clean the data: lower_case all words, remove words with less than one character, just keep alphabet/
#not stopwords and punctuationsas words anymore/ 
feedback_data_all['Feedback_clean'] = feedback_data_all['Feedback_clean'] .apply(lambda x: clean_text(x))
feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback_clean'] == ' ') | (feedback_data_all['Feedback_clean'] == '')].index , inplace=True)
feedback_data_all=feedback_data_all.dropna(subset=['Feedback_clean'])
feedback_data_all=feedback_data_all.reset_index(drop=True)
#---------
max_ner_count_corpus=max(ner_count_feedback) # 17; min=0
feedback_data_all['Norm_NER_count']= feedback_data_all['ner_count_feedback'] / feedback_data_all['word_count']
#np.count_nonzero(feedback_data_all['Norm_NER_count'])=3051,// zeros=10786  
feedback_data_all['pos_count']= feedback_data_all['POS'].apply(lambda x:Counter([j for i,j in x]))  
feedback_data_all['total_pos_count'] = feedback_data_all['pos_count'].apply(lambda x:sum(x.values()))   
count_propernoun =feedback_data_all['pos_count']. apply(lambda x: count_nouns(x))
feedback_data_all['Norm_count_propernoun'] = count_propernoun / feedback_data_all['word_count']    
#np.count_nonzero(feedback_data_all['Norm_count_propernoun'])=6322
count_determiner =feedback_data_all['pos_count']. apply(lambda x: count_derterminer(x))
feedback_data_all['Norm_count_determiner'] = count_determiner / feedback_data_all['word_count'] 
#np.count_nonzero(feedback_data_all['Norm_count_determiner'])=12795
count_wh_det =feedback_data_all['pos_count']. apply(lambda x: count_wh_det(x))
feedback_data_all['Norm_count_wh_det'] = count_wh_det / feedback_data_all['word_count'] 
#np.count_nonzero(feedback_data_all['Norm_count_wh_det'])=6537
count_listM =feedback_data_all['pos_count']. apply(lambda x: count_listM(x)) #there is none in data i guess
feedback_data_all['Norm_count_listM']= count_listM /feedback_data_all['word_count'] 
count_numbers =feedback_data_all['pos_count']. apply(lambda x: count_num(x))
feedback_data_all['Norm_count_num']= count_numbers /feedback_data_all['word_count']
#np.count_nonzero(feedback_data_all['Norm_count_num'])=2087
count_possessive_ending=feedback_data_all['pos_count']. apply(lambda x: count_possessive_end(x))
feedback_data_all['Norm_count_poss_end']= count_possessive_ending /feedback_data_all['word_count']
#np.count_nonzero(feedback_data_all['Norm_count_poss_end'])=666

##------------------plot histogram just to check features-------------------------
#propernoun count
plot_hist(feedback_data_all['Norm_count_propernoun'],
          'Histogram of Normalized Number of Propernoun tags')
# determiner count  (kinda bi-modal)
plot_hist(feedback_data_all['Norm_count_determiner'],
          'Histogram of Normalized Number of Determiners tags')
# w-h count
plot_hist(feedback_data_all['Norm_count_wh_det'],
          'Histogram of Normalized Number of Wh counts tags')
#numbers
plot_hist(feedback_data_all['Norm_count_num'],
          'Histogram of Normalized Number of number tags')
#possessive endings
plot_hist(feedback_data_all['Norm_count_poss_end'],
          'Histogram of Normalized Number of possessive endings tags')
#plot NER histogram
plot_hist(feedback_data_all['Norm_NER_count'],
          'Histogram of Normalized Number of NER tags') #12453 

##lemmatization to substitute words with same root ----------------------------------------------
feedback_data_all['Feedback_clean']= feedback_data_all['Feedback_clean'].apply(lambda x:lemmatize_with_postag(x))
#-----------------plot wordcloud--------------------------------------
show_wordcloud(feedback_data_all['Feedback_clean'])
#-----------------render NER for a random feedback--------------------
displacy.render(nlp(str(feedback_data_all['Feedback_clean'][9179])), jupyter=True, style='ent')   
displacy.render(nlp(str(feedback_data_all['Feedback_clean'][9179])), style='dep', jupyter = True, options = {'distance': 120}) 
##--------------------term frequency-----------------------
array_feedbacks= feedback_data_all['Feedback_clean'].values.astype('U')
vectorizer = CountVectorizer()
count_vectors= vectorizer.fit_transform(array_feedbacks)
print(count_vectors.toarray())
freq = count_vectors.sum(axis=0)
words_freq = [(word, freq[0, idx]) for word, idx in  vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

list_word_rank=[]
list_word_freq=[]
list_word_names=[]
for i in range(0,len(words_freq)):  
    list_word_rank.append(i+1)
    list_word_freq.append((words_freq[i])[1])
    list_word_names.append((words_freq[i])[0])
#plot frequency of word occurence
plot_word_frequency(list_word_rank,list_word_freq,list_word_names)
#plot non-logarithm scatter plot of term frequencies
plot_nonlog_word_frequency(list_word_rank,list_word_freq)

num_unique_words=len(list_word_rank) #5771 unique words
max_freq=max(list_word_freq) #3216
min_freq=min(list_word_freq) #1

##-----------------------sentiment analysis-----------------------------------
sid = SentimentIntensityAnalyzer()
feedback_data_all['sentiments'] = feedback_data_all['Feedback'].apply(lambda x: sid.polarity_scores(str(x)))

list_compound_sent=[]
list_pos_sent=[]
list_neu_sent=[]
list_neg_sent=[]
for i in range(0,len(feedback_data_all['sentiments'])):
    for key, value in (list(feedback_data_all['sentiments'])[i]).items():
        if key == 'neg':
            list_neg_sent.append(value)
        if key == 'neu':
            list_neu_sent.append(value)
        if key == 'pos':
            list_pos_sent.append(value)  
        if key == 'compound':   
            list_compound_sent.append(value) 
feedback_data_all['sentiments_compound']=list_compound_sent
feedback_data_all['sentiments_pos']=list_pos_sent
feedback_data_all['sentiments_neutral']=list_neu_sent
feedback_data_all['sentiments_negative']=list_neg_sent
#------------------------------------    
#Extracting sentiment of the sentence 
feedback_data_all['sentiment_score']=np.zeros(len(feedback_data_all))
max_sent_score=feedback_data_all[['sentiments_pos','sentiments_neutral','sentiments_negative']].idxmax(axis=1)
for i in range (0,len(feedback_data_all['Feedback'])):
         if (max_sent_score[i]=='sentiments_pos') :
              feedback_data_all['sentiment_score'][i] = 'positive'
         elif ((max_sent_score[i]=='sentiments_negative')):
              feedback_data_all['sentiment_score'][i] = 'negative'
         elif ((max_sent_score[i]=='sentiments_neutral')):
              feedback_data_all['sentiment_score'][i] = 'neutral'
#--------------------------------------------------------
#sentiment bar plot
Sentiment_count=feedback_data_all.groupby('sentiment_score').count()
feedback_data_all['sentiment_score'].value_counts(normalize = True)
#negative   1015,  neutral  2828,  positive   10056
draw_sentiment_barplot(Sentiment_count.index.values,Sentiment_count['id'])
# plot sentiment distribution for positive, negative, and neutral feedbacks
sentiment_dataframe=pd.concat([feedback_data_all['sentiment_score'],feedback_data_all['sentiments_compound']],axis=1)
plot_sentimentscore_frequency(sentiment_dataframe)
# to check most positive or most negative feedbacks
feedback_data_all[feedback_data_all['word_count'] > 5].sort_values("sentiments_pos", ascending = False)[["Feedback", "sentiments_pos"]].head(10)
feedback_data_all[feedback_data_all['word_count'] >= 5].sort_values("sentiments_negative", ascending = False)[["Feedback", "sentiments_negative"]].head(10)
feedback_data_all[feedback_data_all['word_count'] >= 5].sort_values("sentiments_neutral", ascending = False)[["Feedback", "sentiments_neutral"]].head(10)    
##--------------------------save the whole dataframe to file--------------------------------
feedback_data_all.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\all_feedback_data.csv', index=False, encoding='utf-8')

##-----------train and test split-use 10% of data for testing------------------
feedback_train_data, feedback_test_data = train_test_split(feedback_data_all,test_size=0.1,random_state=1234)
print("Number of observations in Training Data ",len(feedback_train_data))  
print("Number of observations in Testing Data ",len(feedback_test_data))   

##-------------------tf-idf -----------------------------------------
#Tf-idf for train set
array_feedbacks_train= feedback_train_data['Feedback_clean'].values.astype('U') # list of feedback
vectorizer = TfidfVectorizer(ngram_range=(1,2))  # create the transform (play with n-gram here; this is uni-gram, can try bi-gram later)
vectorizer.fit(array_feedbacks_train) # tokenize and build vocabulary
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
tfidf_vector = vectorizer.transform(array_feedbacks_train)
print(tfidf_vector.shape)
print(tfidf_vector.toarray())
tf_idf_matrix_train= pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names()) #tfidf to dataframe
feedback_index = [n for n in array_feedbacks_train]
feature_names = vectorizer.get_feature_names()
tfidf_matrix_toshow = pd.DataFrame(tfidf_vector.T.todense(), index=feature_names, columns=feedback_index)
print(tfidf_matrix_toshow)
#Tf-idf for test set-----------
array_feedbacks_test= feedback_test_data['Feedback_clean'].values.astype('U') # list of feedback
vectorizer = TfidfVectorizer(ngram_range=(1,2)) 
vectorizer.fit(array_feedbacks_test) 
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
tfidf_vector = vectorizer.transform(array_feedbacks_test)
print(tfidf_vector.shape)
print(tfidf_vector.toarray())
tf_idf_matrix_test= pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names()) 
feedback_index = [n for n in array_feedbacks_test]
feature_names = vectorizer.get_feature_names()
#save tf_idf to use later for modelling and testing
tf_idf_matrix_train.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\TF_IDF_train.csv', index=False, encoding='utf-8')
tf_idf_matrix_test.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\TF_IDF_test.csv', index=False, encoding='utf-8')

##-------------------Doc2Vec------------------------------------------
# create doc2vec vector columns for Train set
documents_train = [TaggedDocument(doc, [i]) for i, doc in enumerate(feedback_train_data['Feedback_clean'].apply(lambda x: x.split(" ")))]
# train a Doc2Vec model text data and transform document into a vector data
model = Doc2Vec(documents_train, vector_size=512, window=2, min_count=1, workers=4) # different vector_size and window_size to be considered
len(model.docvecs) #12517
#train doc2vec model and save it to use for inferring other feedback's doc2vec matrix (for future prediction in webapp)
model.save('doc2vec_saved') 
doc2vec_feedback_train = feedback_train_data['Feedback_clean'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_feedback_train.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_feedback_train.columns]
# create doc2vec vector columns for Test set
documents_test = [TaggedDocument(doc, [i]) for i, doc in enumerate(feedback_test_data['Feedback_clean'].apply(lambda x: x.split(" ")))]
model = Doc2Vec(documents_test, vector_size=512, window=2, min_count=1, workers=4)
doc2vec_feedback_test = feedback_test_data['Feedback_clean'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_feedback_test.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_feedback_test.columns]
#save doc2vec to use later for modelling and testing
doc2vec_feedback_train.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\doc2vec_train.csv', index=False, encoding='utf-8')
doc2vec_feedback_test.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\doc2vec_test.csv', index=False, encoding='utf-8')

##--------------------------save train set and test set with built feature sets to file--------------------------------
#train data
feedback_train_data = pd.concat([feedback_train_data,doc2vec_feedback_train], axis=1)
feedback_train_data = pd.concat([feedback_train_data,tf_idf_matrix_train], axis=1)
feedback_train_data.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\train_data.csv', index=False, encoding='utf-8')
#test data
feedback_test_data = pd.concat([feedback_test_data,doc2vec_feedback_test], axis=1)
feedback_test_data = pd.concat([feedback_test_data,tf_idf_matrix_test], axis=1)
feedback_test_data.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\test_data.csv', index=False, encoding='utf-8')
##--------------------------------------------------------------------------

###------------------Pre-processing functions ----------------------------------
def spell_check (this_feedback):
    '''Check and correct spelling'''    
    words=word_tokenize(str(this_feedback))
    words = [spell.correction(w) for w in words]
    return " ".join(w for w in words) 
#---------------------------------------------------- 
def avg_num_characters(this_feedback):
     '''Computing avg. number of word characters in each text document'''
     words=word_tokenize(str(this_feedback))
     if (len(words) != 0):
         return (sum(len(word) for word in words)/len(words))
     else:
        return 0  # just to not have inf and then delete the whole row
#---------------------------------------------------- 
def is_english_word(word):
    '''function to identify whether a word is english or not''' 
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False
#---------------------------------------------------- 
def remove_nonEnglish(this_feedback):
    '''removing non-english words'''    
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if is_english_word(w)]
    return " ".join(w for w in words) 
#----------------------------------------------------     
def first_clean(this_feedback):
    '''First-step cleaning of the text document'''   
    this_feedback=remove_punctuation(str(this_feedback))
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if (len(w) > 0)] #filter out short tokens (this will delete I and punctuations as well)
    words = [w.lower() for w in words]
    return " ".join(w for w in words)
 #---------------------------------------------------- 
def  clean_text(this_feedback):
     '''Cleaning the text document'''
     this_feedback=remove_stopwords(this_feedback)
     words=word_tokenize(str(this_feedback))
     words = [w for w in words if w.isalpha()] # filters out ? ! , . 've 'r and numbers
     words = [w for w in words if (len(w) > 1 or w == 'i')] #filter out short tokens (this will delete I and punctuations as well)
     return " ".join(w for w in words) 
#---------------------------------------------------- 
def ner_feedback(this_feedback):
    '''Extracting Name Entity Recognition'''
    feedbackk_ents=nlp(str(this_feedback)).ents
    return [x.label_ for x in feedbackk_ents]
#----------------------------------------------------        
def remove_punctuation(this_feedback):
    '''removing punctuations'''
    replace_punct = str.maketrans('', '', string.punctuation) # replacing the punctuations with no space 
    return this_feedback.translate(replace_punct) 
#---------------------------------------------------- 
def remove_stopwords(this_feedback):
    '''removing stop-words''' 
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if not w in stop_words]
    return " ".join(w for w in words) 
#---------------------------------------------------- 
def lemmatize_with_postag(this_feedback):
    """Lemmitization with POS tag (Spacy does this by default)"""
    parsed_feedback= nlp(this_feedback) # Parse the sentence (tokenizing and pos tagging)
    return " ".join([token.lemma_ for token in parsed_feedback]) # Extract the lemma for each token and join
#---------------------------------------------------- 
#---------------------------------------------------- 
def part_speech(this_feedback):
    '''extracting POs tags'''
    words=word_tokenize(str(this_feedback))
    #nltk.ne_chunk(words)  # dont know why this doesnt work, seems like a NLTK bug
    return nltk.pos_tag(words)    
##---------get POS (part-of-speech) count in each category##--------------------
def  count_nouns(pos_count):   #micheal,..
    '''Counting pro-Nouns'''
    sum_pron=0
    for a, b in pos_count.items():
        if a in ('NNP','NNPS'):
        #if (a == 'NNP'or a == 'NNPS'):
            sum_pron=sum_pron+b
    return sum_pron
#----------------------------------------------------         
def  count_derterminer(pos_count):#his/her/that/ my/ his/eithe/both/any
    '''Counting determiners'''
    sum_det=0
    for a, b in pos_count.items():
        if a in ('PRP', 'PRP$','DT'):
            sum_det=b+sum_det
    return  sum_det
#----------------------------------------------------  
def  count_wh_det(pos_count): #what/whose/when?
    '''Counting wh-determiners'''
    sum_wh=0
    for a, b in pos_count.items():
        if a in ('WDT', 'WP','WP$', 'WRB'):
            sum_wh=sum_wh+b
    return sum_wh
#----------------------------------------------------  
def  count_listM(pos_count): #listMaker 1)...
    '''Counting List-Makers'''
    sum_list=0
    for a, b in pos_count.items():
        if a == 'LS':
            sum_list=sum_list+b
    return sum_list
#---------------------------------------------------- 
def  count_num(pos_count): #numbers: 1,2, one, two,....
    '''Counting Numbers'''
    sum_num=0
    for a, b in pos_count.items():
        if a == 'CD':
            sum_num=sum_num+b
    return sum_num
#---------------------------------------------------- 
def  count_possessive_end(pos_count): #mary's
    '''Counting possessive ends'''
    sum_poss_end=0
    for a, b in pos_count.items():
        if a == 'POS':
            sum_poss_end=sum_poss_end+b
    return sum_poss_end
###----------------------------------------------------
