# -*- coding: utf-8 -*-
'''This .py file contains cleaning and pre-processing functions for text documents'''

def spell_check (this_feedback):
    '''Check and correct spelling'''
    from nltk.tokenize import word_tokenize
    from spellchecker import SpellChecker
    
    spell = SpellChecker()
    words=word_tokenize(str(this_feedback))
    words = [spell.correction(w) for w in words]
    return " ".join(w for w in words) 
#---------------------------------------------------- 
def avg_num_characters(this_feedback):
     '''Computing avgerage number of word characters in each text document'''
     from nltk.tokenize import word_tokenize
     
     words=word_tokenize(str(this_feedback))
     if (len(words) != 0):
         return (sum(len(word) for word in words)/len(words))
     else:
        return 0  # just to not have inf and then delete the whole row
#---------------------------------------------------- 
def is_english_word(word):
    '''identify whether a word is english or not'''
    from nltk.corpus import words as nltk_words
    ##English vocabulary;
    #myDict=wordnet.words()
    #dict_words = words.words()
    dictionary = dict.fromkeys(nltk_words.words(), None)  
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False
#---------------------------------------------------- 
def remove_nonEnglish(this_feedback):
    '''removing non-english words'''
    from nltk.tokenize import word_tokenize
    
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if is_english_word(w)]
    return " ".join(w for w in words) 
#----------------------------------------------------     
def first_clean(this_feedback):
    '''First-step cleaning of the text document'''
    from nltk.tokenize import word_tokenize
    
    this_feedback=remove_punctuation(str(this_feedback))
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if (len(w) > 0)] #filter out short tokens (this will delete I and punctuations as well)
    words = [w.lower() for w in words]
    return " ".join(w for w in words)
 #---------------------------------------------------- 
def  clean_text(this_feedback):
     '''Cleaning the text document'''
     from nltk.tokenize import word_tokenize
     
     this_feedback=remove_stopwords(this_feedback)
     words=word_tokenize(str(this_feedback))
     words = [w for w in words if w.isalpha()] # filters out ? ! , . 've 'r and numbers
     words = [w for w in words if (len(w) > 1 or w == 'i')] #filter out short tokens (this will delete I and punctuations as well)
     return " ".join(w for w in words) 
#---------------------------------------------------- 
def ner_feedback(this_feedback):
    '''Extracting Name Entity Recognition'''
    import spacy
    
    nlp = spacy.load('en_core_web_sm')
    feedbackk_ents=nlp(str(this_feedback)).ents
    return [x.label_ for x in feedbackk_ents]
#----------------------------------------------------        
def remove_punctuation(this_feedback):
    '''removing punctuations'''
    import string
    
    replace_punct = str.maketrans('', '', string.punctuation) # replacing the punctuations with no space 
    return this_feedback.translate(replace_punct) 
#---------------------------------------------------- 
def remove_stopwords(this_feedback):
    '''removing stop-words'''
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords 
    import spacy
	
    stopwordss = set(stopwords.words('english'))
    nlp = spacy.load('en_core_web_sm')
    stop_words=(nlp.Defaults.stop_words).union(stopwordss)
    words=word_tokenize(str(this_feedback))
    words = [w for w in words if not w in stop_words]
    return " ".join(w for w in words) 
#---------------------------------------------------- 
def lemmatize_with_postag(this_feedback):
    """Lemmitization with POS tag (Spacy does this by default)"""
    import spacy
    
    nlp = spacy.load('en_core_web_sm')
    parsed_feedback= nlp(this_feedback) # Parse the sentence (tokenizing and pos tagging)
    return " ".join([token.lemma_ for token in parsed_feedback]) # Extract the lemma for each token and join
#---------------------------------------------------- 
#---------------------------------------------------- 
def part_speech(this_feedback):
    '''extracting POs tags'''
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    
    words=word_tokenize(str(this_feedback))
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


