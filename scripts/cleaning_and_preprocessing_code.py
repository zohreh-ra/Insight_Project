# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:41:51 2019

@author: Bahareh
"""
import string
import spacy
import gensim
import nltk
import os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import words
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob, Word 
from collections import Counter
from nltk.tag import pos_tag
from nltk import FreqDist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import seaborn as sns
from spacy import displacy


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) #keeping only tagger component needed for lemmatization
#apppend stopwords from different libraries (slightly different in each library; to have a more comprehensive coverage)
stopwordss = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
stop_words=(nlp.Defaults.stop_words).union(stopwordss)
#English vocabulary;
myDict=wordnet.words()
dict_words = words.words()

# load data
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
feedback_data_all= pd.read_csv('FeedbacksSpecificityTone.csv', encoding='utf-8',header=None, names=["id", "Feedback"])
feedback_data_all.drop_duplicates(subset ="Feedback", keep = 'first', inplace = True)
feedback_data_all=feedback_data_all.reset_index(drop=True)
 
# removing non-english words and blank rows and correct spelling of words
feedback_data_all['Feedback'] = feedback_data_all['Feedback'].apply(lambda x: spell_check(x))  
feedback_data_all['Feedback'] = feedback_data_all['Feedback'].apply(lambda x: remove_nonEnglish(x))  #does not work , removes many words (better to give it a sentence rther than word by word)
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
#sttopword count
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
n, bins, patches = plt.hist(x=feedback_data_all['Norm_count_propernoun'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normalized Number of Propernoun tags')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# determiner count  (kinda bi-modal)
n, bins, patches = plt.hist(x=feedback_data_all['Norm_count_determiner'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normalized Number of Determiners tags')
plt.text(23, 45, r'$\mu=15, b=3$')
# w-h count
n, bins, patches = plt.hist(x=feedback_data_all['Norm_count_wh_det'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normalized Number of Wh counts tags')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
#numbers
n, bins, patches = plt.hist(x=feedback_data_all['Norm_count_num'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normalized Number of number tags')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
#possessive endings
feedback_data_all['Norm_count_poss_end']
n, bins, patches = plt.hist(x=feedback_data_all['Norm_count_poss_end'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normalized Number of possessive endings tags')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
#plot NER histogram
n, bins, patches = plt.hist(x=feedback_data_all['Norm_NER_count'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normalized Number of NER tags')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()   #12453 

##lemmatization to get words with same root are same----------------------------------------------
feedback_data_all['Feedback_clean']= feedback_data_all['Feedback_clean'].apply(lambda x:lemmatize_with_postag(x))
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
num_unique_words=len(list_word_rank) #5771 unique words
max_freq=max(list_word_freq) #3216
min_freq=min(list_word_freq) #1

##-------------------tf idf  (this is uni-gram, can try bi-gram later)---------------
array_feedbacks= feedback_data_all['Feedback_clean'].values.astype('U') # list of feedbacks
vectorizer = TfidfVectorizer(ngram_range=(1,2))  # create the transform (play with n-gram here)
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
print(tfidf_matrix_toshow)

##-----------------------sentiment analysis-----------------------------------
sid = SentimentIntensityAnalyzer()
feedback_data_all['sentiments'] = feedback_data_all['Feedback'].apply(lambda x: sid.polarity_scores(x))

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
#positive %    0.723505 , neutral%   0.203468, negative%    0.073027  #compound threshold=0.05
#positive    0.699885, neutral     0.240437, negative    0.059678   #compound threshold=0.2
#neutral     0.915085; positive    0.080673;negative    0.004242 #when go with percentage of positive/neg/neutral
plt.bar(Sentiment_count.index.values, Sentiment_count['id'])
plt.xlabel('Feedback Sentiment')
plt.ylabel('Number of Feedbacks')
plt.show()

# plot sentiment distribution for positive, negative, and neutral feedbacks
for sentiment in ['negative','positive','neutral']:
    subset = feedback_data_all[feedback_data_all['sentiment_score'] == sentiment]   
    # Draw the density plot
    if sentiment == 'negative':
        label_plot = "Bad Feedbacks"
    elif sentiment == 'positive':
        label_plot = "Good Feedbacks"
    else:   
        label_plot = "Neutral Feedbacks"
    sns.distplot(subset['sentiments_compound'], hist = False, label = label_plot) 

# to check most positive or most negative feedbacks
feedback_data_all[feedback_data_all['word_count'] > 5].sort_values("sentiments_pos", ascending = False)[["Feedback", "sentiments_pos"]].head(10)
feedback_data_all[feedback_data_all['word_count'] >= 5].sort_values("sentiments_negative", ascending = False)[["Feedback", "sentiments_negative"]].head(10)
feedback_data_all[feedback_data_all['word_count'] >= 5].sort_values("sentiments_neutral", ascending = False)[["Feedback", "sentiments_neutral"]].head(10)
    
##-------------------Doc2Vec------------------------------------------
# create doc2vec vector columns
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(feedback_data_all['Feedback_clean'].apply(lambda x: x.split(" ")))]
# train a Doc2Vec model text data and transform document into a vector data
model = Doc2Vec(documents, vector_size=512, window=2, min_count=1, workers=4)
doc2vec_feedback = feedback_data_all['Feedback_clean'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_feedback.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_feedback.columns]
feature_set= pd.concat([feature_set, doc2vec_feedback], axis=1)
##--------------------------save the whole dataframe to file--------------------------------
feedback_data_all = pd.concat([feedback_data_all,doc2vec_feedback.reset_index(drop=True)], axis=1)
feedback_data_all = pd.concat([feedback_data_all,tf_idf_matrix.reset_index(drop=True)], axis=1)
feedback_data_all.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\all_feedback_data.csv', index=False, encoding='utf-8')
##----------------preparing feature set-----------------------------------------------------
#save tf_idf to use later for modelling
tf_idf_matrix.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\TF_IDF.csv', index=False, encoding='utf-8')
#save doc2vec to use later for modelling
doc2vec_feedback.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\doc2vec.csv', index=False, encoding='utf-8')
##-----------train and test split-use 10% of data for testing------------------
feedback_train_data, feedback_test_data = train_test_split(feedback_data_all,test_size=0.1,random_state=1234)
print("Number of observations in Training Data ",len(feedback_train_data))  #12517
print("Number of observations in Testing Data ",len(feedback_test_data))   #1391
feedback_test_data.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\Test_data3.csv', index=False, encoding='utf-8')
feedback_train_data.to_csv(r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data\Train_data3.csv', index=False, encoding='utf-8')
##--------------------------------------------------------------------------
#plot wordcloud
show_wordcloud(feedback_data_all['Feedback_clean'])
#-------------------------------------------------------------------------------

def is_english_word(word):
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False

def remove_nonEnglish(this_feedback):
    '''removing non-english words'''
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if is_english_word(w)]
    return " ".join(w for w in words) 

def spell_check (this_feedback):
    '''Check and correct spelling'''
    words=word_tokenize(str(this_feedback))
    words = [spell.correction(w) for w in words]
    return " ".join(w for w in words) 

def avg_num_characters(this_feedback):
     '''Computing avg. number of characters in each text feedback'''
     words=word_tokenize(str(this_feedback))
     if (len(words) != 0):
         return (sum(len(word) for word in words)/len(words))
     else:
        return 0  # just to not have inf and then delete the whole row
    
def first_clean(this_feedback):
    this_feedback=remove_punctuation(str(this_feedback))
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if (len(w) > 0)] #filter out short tokens (this will delete I and punctuations as well)
    words = [w.lower() for w in words]
    return " ".join(w for w in words)
 
def  clean_text(this_feedback):
     '''Cleaning the text document'''
     this_feedback=remove_stopwords(this_feedback)
     words=word_tokenize(str(this_feedback))
     words = [w for w in words if w.isalpha()] # filters out ? ! , . 've 'r and numbers
     words = [w for w in words if (len(w) > 1 or w == 'i')] #filter out short tokens (this will delete I and punctuations as well)
     return " ".join(w for w in words) 

def ner_feedback(this_feedback):
    '''Extracting Name Entity Recognition'''
    feebbackk_ents=nlp(str(this_feedback)).ents
    return [x.label_ for x in feebbackk_ents]
       
def part_speech(this_feedback):
    '''extracting POs tags'''
    words=word_tokenize(str(this_feedback))
    #nltk.ne_chunk(words)  # dont know why this doesnt work, seems like a NLTK bug
    return nltk.pos_tag(words)    

def remove_punctuation(this_feedback):
    '''removing punctuations'''
    replace_punct = str.maketrans('', '', string.punctuation) # replacing the punctuations with no space 
    return this_feedback.translate(replace_punct) 

def remove_stopwords(this_feedback):
    '''removing stop-words'''
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if not w in stop_words]
    return " ".join(w for w in words) 

def lemmatize_with_postag(this_feedback):
    """Lemmitization with POS tag (Spacy does this by default)"""
    parsed_feedback= nlp(this_feedback) # Parse the sentence (tokenizing and pos tagging)
    return " ".join([token.lemma_ for token in parsed_feedback]) # Extract the lemma for each token and join

#get pos count in each category#--------------------
def  count_nouns(pos_count):   #micheal,..
    sum_pron=0
    for a, b in pos_count.items():
        if a in ('NNP','NNPS'):
        #if (a == 'NNP'or a == 'NNPS'):
            sum_pron=sum_pron+b
    return sum_pron
        
def  count_derterminer(pos_count):#his/her/that/ my/ his/I/ she/he
    sum_det=0
    for a, b in pos_count.items():
        if a in ('PRP', 'PRP$','DT'):
            sum_det=b+sum_det
    return  sum_det
 
def  count_wh_det(pos_count): #what/whose/when
    sum_wh=0
    for a, b in pos_count.items():
        if a in ('WDT', 'WP','WP$', 'WRB'):
            sum_wh=sum_wh+b
    return sum_wh
 
def  count_listM(pos_count): #listMaker 1)...
    sum_list=0
    for a, b in pos_count.items():
        if a == 'LS':
            sum_list=sum_list+b
    return sum_list

def  count_num(pos_count): #listMaker 1)...
    sum_num=0
    for a, b in pos_count.items():
        if a == 'CD':
            sum_num=sum_num+b
    return sum_num

def  count_possessive_end(pos_count): #listMaker 1)...
    sum_poss_end=0
    for a, b in pos_count.items():
        if a == 'POS':
            sum_poss_end=sum_poss_end+b
    return sum_poss_end
  #-------------------------------------------------      
# wordcloud function
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'black', max_words = 200, max_font_size = 40, scale = 3,
        random_state = 42).generate(str(data))
    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)
    plt.imshow(wordcloud)
    plt.show()
#----------------------------------------------------
