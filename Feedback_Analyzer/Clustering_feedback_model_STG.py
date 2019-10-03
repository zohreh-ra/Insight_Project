# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:41:51 2019

@author: Bahareh
"""

"""this code aims at seperating "Vague" and "Specific" feedback of teachers
 using unsupervised learning technique"""

import numpy as np, pandas as pd
import os
import nltk
import spacy
import gensim
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from textblob import TextBlob, Word
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering 
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.manifold import MDS 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import time
import pickle

# load data
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
feedback_data= pd.read_csv('Train_data3.csv', encoding='utf-8')
#features_doctovec=pd.read_csv('doc2vec.csv', encoding='utf-8')
#features_tfidf=pd.read_csv('TF_IDF.csv', encoding='utf-8')

# removing  blank rows; in case
feedback_data.drop(feedback_data[(feedback_data['Feedback'] == ' ') | (feedback_data['Feedback'] == '') ].index , inplace=True)
feedback_data=feedback_data.reset_index(drop=True)

##--------------------Feedback clustering-------------------------------- 
##------preparing feature set --------------------
#play with features and run the model, use the best set of features eventually
#one potential feature: sentiment compound score 
feature_pos=pd.concat([feedback_data['Norm_NER_count'],feedback_data['Norm_count_propernoun'],feedback_data['Norm_count_wh_det'],feedback_data['Norm_count_determiner'],feedback_data['Norm_avg_num_characters'],feedback_data['normalized_stopwords_count'],feedback_data['Norm_count_poss_end'],feedback_data['Norm_count_num'],feedback_data['Norm_uppercase_count'],feedback_data['sentiments_compound']], axis=1)  
feature_pos=pd.concat([feedback_data[feedback_data.columns[30:541]],feature_pos],axis=1) #adding Doc2Vec Matrix to features
#feature_set = StandardScaler().fit_transform(feature_set) #my data is already normalized
##-----------------------------------------
##--------- K-Means clustering--------------
km = KMeans(n_clusters=2,max_iter=3000)
kmeans =km.fit(feature_pos)
clusters = kmeans.labels_.tolist()
centroids = km.cluster_centers_
labels = kmeans.predict(feature_pos)
Counter(clusters) #Counter({0: 9876, 1: 2641})

##------- hierarchical clustering--------
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage='ward')
y_hc=hc.fit_predict(feature_pos)
clusters = hc.labels_.tolist()
Counter(clusters)   # Counter({0: 10210, 1: 2307})
#plot dendrogram --example-----
dendrogram = sch.dendrogram(sch.linkage(feature_pos, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
##------------GMM clustering-----------------------------
gmm = GMM(n_components=2,max_iter=400).fit(feature_pos)
clusters = gmm.predict(feature_pos)
probs = gmm.predict_proba(feature_pos)
Counter(clusters) #Counter({0: 10512, 1: 2005})

##------------Special clustering---------------------------
clustering = SpectralClustering(n_clusters=2,assign_labels="discretize", random_state=0).fit(feature_pos)
clusters=clustering.labels_.tolist()
Counter(clusters) # Counter({0: 9799, 1: 2718})

##------------Plot clusters--------------------
#dimentionality reduction pca 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(feature_pos)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#set up colors and cluster names per clusters 
cluster_colors = {0: '#1b9e77', 1: '#d95f02'}
cluster_names = {0: 'Specific Feedback', 1: 'Vague Feedback'}

xs=principalDf.iloc[:, 0].tolist()
ys=principalDf.iloc[:, 1].tolist()
#-----------plot using PCA-------------------
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g']
targets=[0,1]

finalDf = pd.concat([principalDf,pd.DataFrame(clusters).reset_index()], axis = 1)

for target, color in zip(targets,colors):
    indicesToKeep = clusters == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#----------------------------------
#----------Plot using TSNE reduction----------
tsne_init = 'pca'  #start with 'pca' or even can use 'random'
tsne_perplexity = 50.0
tsne_early_exaggeration = 4.0
tsne_learning_rate = 1000
random_state = 1
model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
         early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate,n_iter=1000)

time_start=time.time()
transformed_points = model.fit_transform(feature_pos)
print('t_SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

print (transformed_points)

tsneDf = pd.DataFrame(data = transformed_points, columns = ['TSNE1', 'TSNE 2'])


cluster_colors = {0: '#1b9e77', 1: '#d95f02'}
cluster_names = {0: 'Specific Feedback', 1: 'Vague Feedback'}

xs=tsneDf.iloc[:, 0].tolist()
ys=tsneDf.iloc[:, 1].tolist()

%matplotlib inline 

title=[]

for i in range(0 , len(feedback_data)):
    first_2words=' '.join(str(feedback_data['Feedback'].iloc[i]).split()[:2])
    title.append(first_2words)
     
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=title)) 
#group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) 

#iterate through groups 
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         
        which='both',     
        left='off',     
        top='off',         
        labelleft='off')
    
ax.legend(numpoints=1)  

for i in range(len(df)):
   ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8) 
plt.show() #show the plot
plt.savefig('clusters_plot.png', dpi=200)

##---------Calculate feature_importance-------
# random forest classifier (after labelling, treat the problem as a supervised learning problem to get feature importance)
randomf = RandomForestClassifier(n_estimators=100,random_state = 42)
randomf.fit(feature_pos, clusters)
# show feature importance
feature_importances_feedback = pd.DataFrame({"feature": feature_pos, "importance": randomf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_feedback.head(20) #20 first important features
s
##---------------use training data and train KNN model and save it for future prediction------------
#train features and labels:  feature_pos, train_labels
train_labels=pd.DataFrame(clusters)
filename='knn_train_model_saved.sav'
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(feature_pos,train_labels.values.ravel())
pickle.dump(model_knn, open(filename, 'wb')) #save the model
#-----------------------------------------------------------------------







