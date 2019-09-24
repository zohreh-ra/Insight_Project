# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:41:51 2019

@author: Bahareh
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import StanfordNERTagger
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from textblob import TextBlob, Word
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.manifold import MDS 
import numpy as np, pandas as pd
from collections import Counter
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
#from sklearn.mixture import GMM
from nltk.corpus import words
from nltk.tag import pos_tag
from nltk import FreqDist
import string
import spacy
import gensims
import nltk
import os
import numpy as np, pandas as pd
from nltk.corpus import words
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize
from nltk.corpus import words as nltk_words
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #keeping only tagger component needed for lemmatization
stopwordss = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
stop_words=(nlp.Defaults.stop_words).union(stopwordss)

#English vocabulary; tried to append two vocabularies but wasnt successful!
myDict=wordnet.words()
myList=words.words()
dict_words = words.words()

# load data
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
feedback_data_all= pd.read_csv('FeedbacksSpecificityTone.csv', encoding='utf-8',header=None, names=["id", "Feedback"])
feedback_data_all.drop_duplicates(subset ="Feedback", keep = 'first', inplace = True)
feedback_data_all=feedback_data_all.reset_index(drop=True)

dictionary = dict.fromkeys(nltk_words.words(), None)   
def is_english_word(word):
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False

def remove_nonEnglish(this_feedback):
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if is_english_word(w)]  #takes forever to run
    return " ".join(w for w in words) 


feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback'] == ' ') | (feedback_data_all['Feedback'] == '') ].index , inplace=True)
feedback_data_all=feedback_data_all.reset_index(drop=True)
#tokenized words for each feedback
feedback_data_all['words_seppunct'] = feedback_data_all['Feedback'].apply(lambda x: word_tokenize(str(x))) #punctuation seperated and space

feedback_data_all['words_sepspace'] = feedback_data_all['Feedback'].apply(lambda x: str(x).split(" "))  # only space seperated
#number of words
feedback_data_all['word_count'] = feedback_data_all['words_seppunct'].apply(lambda x: len(x))  # not to include punctuations in word count
#number of characters
feedback_data_all['char_count'] = feedback_data_all['Feedback'].str.len() ## this also includes spaces
#avg number of characters in the words of a feedback= average word length
feedback_data_all['avg_num_characters'] = feedback_data_all['Feedback'].apply(lambda x: avg_num_characters(x))
#number of stopwords and normalized number of stop words
stopwords_count = feedback_data_all['Feedback'].apply(lambda x: len([x for x in word_tokenize(str(x)) if x in stop_words]))
feedback_data_all['normalized_stopwords_count'] = stopwords_count / feedback_data_all['word_count']
#number of uppercase words
feedback_data_all['uppercase'] = feedback_data_all['Feedback'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))

#lower_case all words, remove words with less than one character, just keep alphabet, stemmizing/
# extract whatever feature that you want to/lowecase=uppercase, lemmatization to get words with same root are the same, not stopwords and punctuationsas words anymore/
# (better to categorze words with the same root as one category when plotting frequency)
feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback'] == ' ') | (feedback_data_all['Feedback'] == '')].index , inplace=True)
feedback_data_all=feedback_data.reset_index(drop=True)
feedback_data_all['Feedback_clean'] = feedback_data_all['Feedback'] .apply(lambda x: clean_text(x))
feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback_clean'] == ' ') | (feedback_data_all['Feedback_clean'] == '')].index , inplace=True)
feedback_data_all.drop(feedback_data_all[(feedback_data_all['Feedback'] == ' ') | (feedback_data_all['Feedback'] == '')].index , inplace=True)
feedback_data_all=feedback_data_all.dropna(subset=['Feedback'])
feedback_data_all=feedback_data_all.dropna(subset=['Feedback_clean'])
feedback_data_all=feedback_data_all.reset_index(drop=True)
feedback_data_all['POS']=np.zeros(len(feedback_data_all))
#---------
 #POS extraction
for i in range(0,len(feedback_data_all['Feedback_clean'])):
         feedback_data_all['POS'].iloc[i]= part_speech(feedback_data_all['Feedback_clean'].iloc[i])         
feedback_data_all['POS'] =feedback_data_all['Feedback_clean']. apply(lambda x: part_speech(x))
feedback_data_all['pos_count']= feedback_data_all['POS'].apply(lambda x:Counter([j for i,j in x]))  
feedback_data_all['total_pos_count'] = feedback_data_all['pos_count'].apply(lambda x:sum(x.values()))      
feedback_data_all['count_propernoun']=feedback_data_all['pos_count']. apply(lambda x: count_nouns(x))
feedback_data_all['count_determiner']=feedback_data_all['pos_count']. apply(lambda x: count_derterminer(x))
feedback_data_all['count_wh_det']=feedback_data_all['pos_count']. apply(lambda x: count_wh_det(x))
feedback_data_all['count_listM']=feedback_data_all['pos_count']. apply(lambda x: count_listM(x)) #there is none in data i guess


feedback_data_all['Feedback_clean']= feedback_data_all['Feedback_clean'].apply(lambda x:lemmatize_with_postag(x))

#term frequency
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
plt.title("Word Frequencies")
plt.ylabel("Log of Total Number of Occurrences")
plt.xlabel("Log of Rank of word")
plt.loglog(list_word_rank,list_word_freq,basex=10,label=list_word_names)
plt.show()

#plot non-logarithm scatter plot of term frequencies
plt.title("Word Frequencies")
plt.ylabel("Total Number of Occurrences")
plt.xlabel("Rank of word")
plt.scatter(list_word_rank,list_word_freq);
plt.show()
num_unique_words=len(list_word_rank) #5205 unique words
max_freq=max(list_word_freq) #34329
min_freq=min(list_word_freq) #1

#tf idf 
array_feedbacks= feedback_data_all['Feedback_clean'].values.astype('U') # list of feedbacks
vectorizer = TfidfVectorizer()  # create the transform (play with n-gram here)
vectorizer.fit(array_feedbacks) # tokenize and build vocab
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
tfidf_vector = vectorizer.transform(array_feedbacks)
print(tfidf_vector.shape)
print(tfidf_vector.toarray())
tf_idf_matrix= pd.DataFrame(tfidf_vector.toarray(), columns=vectorizer.get_feature_names()) #tfidf to dataframe
feedback_index = [n for n in array_feedbacks]
feature_names = vectorizer.get_feature_names()
tfidf_matrix_toshow = pd.DataFrame(tfidf_vector.T.todense(), index=feature_names, columns=feedback_index)
print(df)

X1 =  pd.concat([feedback_data_all['count_propernoun'],feedback_data_all['count_wh_det'],feedback_data_all['avg_num_characters'],feedback_data_all['normalized_stopwords_count']], axis=1)  #add NER later
feature_set = pd.concat([X1.reset_index(drop=True),tf_idf_matrix.reset_index(drop=True)], axis=1)  #concat feature columns
#no feedback_data_all['count_listM'] in data
# why feedback_data_all['count_propernoun']=0 for all data, it shows up like -pron- but nit counted

#sentiment analysis (on feednack_data[feedback])
sid = SentimentIntensityAnalyzer()
feedback_data_all['sentiments'] = feedback_data_all['Feedback'].apply(lambda x: sid.polarity_scores(x))
#feedback_data_all['sent_score'] = pd.cut(feedback_data_all['sentiments'], bins=5, labels=[1, 2, 3, 4, 5]) 

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
    
feedback_data_all['sentiment_score']=np.zeros(len(feedback_data_all))
for i in range (4000,len(feedback_data_all['Feedback'])):
         if ((feedback_data_all['sentiments'][i])['compound'] >= 0.05):
          feedback_data_all['sentiment_score'][i] = 'positive'
         elif ((feedback_data_all['sentiments'][i])['compound']  <= -0.05):
              feedback_data_all['sentiment_score'][i] = 'negative'
         else:
              feedback_data_all['sentiment_score'][i] = 'neutral'

#sentiment bar plot
Sentiment_count=feedback_data_all.groupby('sentiment_score').count()
feedback_data_all['sentiment_score'].value_counts(normalize = True)
#negative   1015,  neutral  2828,  positive   10056
#positive %    0.723505 , neutral%   0.203468, negative%    0.073027
plt.bar(Sentiment_count.index.values, Sentiment_count['id'])
plt.xlabel('Feedback Sentiments')
plt.ylabel('Number of Feedback')
plt.show()

feature_set = pd.concat([feature_set, feedback_data_all['sentiments']], axis=1)

# create doc2vec vector columns
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(feedback_data_all['Feedback_clean'].apply(lambda x: x.split(" ")))]
# train a Doc2Vec model with data and transform each document into a vector data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
doc2vec_feedback = feedback_data_all['Feedback_clean'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_feedback.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_feedback.columns]
feature_set= pd.concat([feature_set, doc2vec_feedback], axis=1)

# wordcloud function
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(feedback_data_all['Feedback_clean'])



# use 1% of data for testing
feedback_train_data, feedback_test_data = train_test_split(feedback_data_all,test_size=0.01,random_state=1234)
print("Number of observations in Training Data ",len(feedback_train_data))
print("Number of observations in Testing Data ",len(feedback_test_data))

feedback_test_data.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\Test_data2.csv', index=False, encoding='utf-8')
feedback_train_data.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\Train_data2.csv', index=False, encoding='utf-8')



def avg_num_characters(this_feedback):
     '''Computing avg. number of characters in each text feedback'''
     words=word_tokenize(str(this_feedback))
     if (len(words) != 0):
         return (sum(len(word) for word in words)/len(words))
     else:
        return 0  # just to not have inf and then delete the whole row
  
def clean_text(this_feedback):
    '''Cleaning the text document'''
     words=word_tokenize(str(this_feedback))
     this_feedback=remove_punctuation(this_feedback)
     this_feedback=remove_stopwords(this_feedback)
     words = [w.lower() for w in words]
     words = [w for w in words if w.isalpha()] # filters out ? ! , . 've 'r and numbers
     words = [w for w in words if len(w) > 1] #filter out short tokens (this will delete I and punctuations as well)
     return " ".join(w for w in words) 

def part_speech(this_feedback):
     '''extracting POs tags'''
    words=word_tokenize(str(this_feedback))
    #nltk.ne_chunk(words)  # dont know why this doesnt work, seems like a NLTK bug
    return nltk.pos_tag(words)    

def remove_punctuation(this_feedback):
    '''removing punctuations'''
    # replacing the punctuations with no space  
    replace_punct = str.maketrans('', '', string.punctuation)
    return this_feedback.translate(replace_punct) 

def remove_stopwords(this_feedback):
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if not w in stop_words]
    return " ".join(w for w in words) 


def lemmatize_with_postag(this_feedback):
    """Lemmitization with POS tag (Spacy do  this by default)"""
    parsed_feedback= nlp(this_feedback) # Parse the sentence (tokenizing and pos tagging)
    return " ".join([token.lemma_ for token in parsed_feedback]) # Extract the lemma for each token and join


#get pos count in each category#--------------------
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
  #-------------------------------------------------      


