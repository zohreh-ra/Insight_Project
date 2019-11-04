# -*- coding: utf-8 -*-
"""
@author: Zohreh
"""
"""this code aims at seperating "Vague" and "Specific" feedback of teachers
 using unsupervised learning technique"""

#!/usr/bin/env python3
import os
import numpy as np, pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering 
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from functions_visualization import plot_two_Histograms
from functions_visualization import plot_clusters_pca
from functions_visualization import plot_explained_variances

# load data
target_workspace =r'C:\Users\Bahareh\Documents\Insight_Program\Sown to Grow project\Data'
os.chdir(target_workspace)
feedback_data= pd.read_csv('Train_data3.csv', encoding='utf-8')
#features_doctovec=pd.read_csv('doc2vec.csv', encoding='utf-8')
#features_tfidf=pd.read_csv('TF_IDF.csv', encoding='utf-8')

# removing  blank rows; (in case)
feedback_data.drop(feedback_data[(feedback_data['Feedback'] == ' ') | (feedback_data['Feedback'] == '') ].index , inplace=True)
feedback_data=feedback_data.reset_index(drop=True)

##------preparing feature set --------------------
#play with features and run the model, use the best set of features eventually
#one potential feature: sentiment compound score 
feature_pos1=pd.concat([feedback_data['Norm_NER_count'],feedback_data['Norm_count_propernoun'],feedback_data['Norm_count_wh_det'],feedback_data['Norm_count_determiner'],feedback_data['Norm_avg_num_characters'],feedback_data['normalized_stopwords_count'],feedback_data['Norm_count_poss_end'],feedback_data['Norm_count_num'],feedback_data['Norm_uppercase_count'],feedback_data['sentiments_compound']], axis=1)  
# calculate explained variance by each PCA-----------
cov_mat = np.cov(feature_pos1)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot_variance = sum(eigen_vals) #  cumulative sum of explained variances
exp_variance= [(i / tot_variance) for i in sorted(eigen_vals, reverse=True)]
cum_exp_variance = np.cumsum(exp_variance)
plot_explained_variances(exp_variance,cum_exp_variance)

#correlation matrix
corr_mat=feature_pos1.corr()
# plot the heatmap of correlation matrix
sns.heatmap(corr_mat,xticklabels=corr_mat.columns,yticklabels=corr_mat.columns)
##-----------------------------------------
#Final selected feature set
feature_pos=pd.concat([feedback_data['Norm_NER_count'],feedback_data['Norm_count_determiner'], feedback_data['normalized_stopwords_count'],feedback_data['Norm_avg_num_characters']], axis=1)  
feature_pos=pd.concat([feature_pos,feedback_data[feedback_data.columns[30:542]]],axis=1) #adding Doc2Vec Matrix to features
#feature_pos = StandardScaler().fit_transform(feature_pos) #data is already normalized

##--------------------Feedback clustering-------------------------------- 
##--------- K-Means clustering--------------
km = KMeans(n_clusters=2,max_iter=3000)
kmeans =km.fit(feature_pos)
clusters = kmeans.labels_.tolist()
centroids = km.cluster_centers_
labels = kmeans.predict(feature_pos)
Counter(clusters)

##------- hierarchical clustering--------
# for final model, Hierarchical clustering was chosen 
#plot dendrogram ---
dist_d=pdist(feature_pos) 
dendrogram = sch.dendrogram(sch.linkage(dist_d, method  = "ward"),labels=list(range(0,len(feature_pos))))
plt.title('Dendrogram')
plt.xlabel('Feedback')
plt.ylabel('Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage='ward')
y_hc=hc.fit_predict(feature_pos)
labels=hc.labels_
clusters = hc.labels_.tolist()
labels=hc.labels_
Counter(clusters)   #Counter({1: 10071, 0: 2446})  #doc2vec,no sentiment, no uppercase,no possessive det count, no num, no proper_noun, no wh_det, no ner, no determiner 
                    #Counter({1: 8014, 0: 4503})   #doc2vec,no sentiment, no uppercase,no possessive det count, no num, no proper_noun, no wh_det, no ner (This one was eventually selected)
                    #Counter({1: 9234, 0: 3283})   #doc2vec,no sentiment, no uppercase,no possessive det count, no num, no proper_noun, no wh_det
                    
#compute Silhoutte score (quality of clustering)
score = silhouette_score (feature_pos, labels, metric='euclidean') #0.278                 
##------------GMM clustering-----------------------------
gmm = GMM(n_components=2,max_iter=4000).fit(feature_pos)
labels = gmm.predict(feature_pos)
clusters = labels.tolist()
Counter(clusters) 
probs = gmm.predict_proba(feature_pos)

##------------Special clustering---------------------------
clustering = SpectralClustering(n_clusters=2,assign_labels="discretize", random_state=0).fit(feature_pos)
clusters=clustering.labels_.tolist()
Counter(clusters) 
##---------------------------------------

##draw histogram of features for each cluster-----------
labels_df=pd.DataFrame(labels)
features_and_labels=pd.concat([feature_pos1,labels_df],axis=1)
features_and_labels.rename(columns={0:'labels'}, inplace=True)
subset_group1=features_and_labels.loc[features_and_labels['labels'] == 0]
subset_group2=features_and_labels.loc[features_and_labels['labels'] == 1]

#propernoun count
plot_two_Histograms(subset_group1['Norm_count_propernoun'],subset_group2['Norm_count_propernoun'],
                        'Histogram Comparison of Normalized Number of Propernoun tags in two clusters',
                        'hist_count_propernoun_hc.png')
#NER count
plot_two_Histograms(subset_group1['Norm_NER_count'],subset_group2['Norm_NER_count'],
                        'Histogram Comparison of Normalized NER Counts in two clusters',
                        'hist2_count_ner_hc.png')
#W-h deterniners count
plot_two_Histograms(subset_group1['Norm_count_wh_det'],subset_group2['Norm_count_wh_det'],
                        'Histogram Comparison of Normalized W-h Determiner Counts in two clusters',
                        'hist2_count_wh_hc.png')
#determiners count
plot_two_Histograms(subset_group1['Norm_count_determiner'],subset_group2['Norm_count_determiner'],
                        'Histogram Comparison of Normalized Determiner Counts in two clusters',
                        'hist2_count_determiner_hc.png')
#avg. num characters
plot_two_Histograms(subset_group1['Norm_avg_num_characters'],subset_group2['Norm_avg_num_characters'],
                        'Histogram Comparison of Average Number of Characters in two clusters',
                        'hist2_avg_num_char_hc.png')
#normalized stopword counts
plot_two_Histograms(subset_group1['normalized_stopwords_count'],subset_group2['normalized_stopwords_count'],
                        'Histogram Comparison of Normalized Stop-word Counts in two clusters',
                        'hist2_count_stopwords_hc.png')
#possession end counts
plot_two_Histograms(subset_group1['Norm_count_poss_end'],subset_group2['Norm_count_poss_end'],
                        'Histogram Comparison of Normalized Count of possessive ends in two clusters',
                        'hist2_count_possession_hc.png')
#Normalized count of numbers
plot_two_Histograms(subset_group1['Norm_count_num'],subset_group2['Norm_count_num'],
                        'Histogram Comparison of Normalized Count of Numbers in two clusters',
                        'hist2_count_num_hc.png')
#Normalized count of uppercase 
plot_two_Histograms(subset_group1['Norm_uppercase_count'],subset_group2['Norm_uppercase_count'],
                        'Histogram Comparison of Normalized Count of Uppercase Words in two clusters',
                        'hist2_count_uppercase_hc.png')
#sentiment score 
plot_two_Histograms(subset_group1['sentiments_compound'],subset_group2['sentiments_compound'],
                        'Histogram Comparison of Sentimnet Scores in two clusters',
                        'hist2_sentiment_hc.png')

##---------------------------
##------------Plot clusters--------------------
#dimentionality reduction pca 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(feature_pos)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf,pd.DataFrame(clusters).reset_index()], axis = 1)

xs=principalDf.iloc[:, 0].tolist()
ys=principalDf.iloc[:, 1].tolist()

plot_clusters_pca(principalDf,labels) #plot using PCA
#----------------------------------
#----------Plot using TSNE reduction----------
tsne_init = 'pca'  #start with 'pca' or even can use 'random'
tsne_perplexity = 50.0
tsne_early_exaggeration = 4.0
tsne_learning_rate = 1000
#random_state = 1
model = TSNE(n_components=2, init=tsne_init, perplexity=tsne_perplexity,
         early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate,n_iter=1000)

time_start=time.time()
transformed_points = model.fit_transform(feature_pos)
print('t_SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
print (transformed_points)

tsneDf = pd.DataFrame(data = transformed_points, columns = ['TSNE1', 'TSNE 2'])

cluster_colors = {0: 'cornflowerblue', 1: '#1b9e77'}
cluster_names = {0: 'Vague Feedback', 1: 'Specific Feedback'}
xs=tsneDf.iloc[:, 0].tolist()
ys=tsneDf.iloc[:, 1].tolist()

%matplotlib inline 

title=[]

for i in range(0 , len(feedback_data)):
    all_sentnce=str(feedback_data['Feedback'].iloc[i])
    title.append(all_sentnce)
       
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

#for i in range(len(df)):
   #ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8) 
   
##--- to print some example output on the plot instead of all that is not readable----------        
#rows_defined=[546,589,3521,4372, 6592]
#text_to_print=["'some feedback text corresponding to rows defined'"]
#for i,j in zip(rows_defined,text_to_print):
   #ax.annotate(j, (df.iloc[i]['x'], df.iloc[i]['y']), size=18) 
  
plt.show() #show the plot
plt.savefig('clusters_plot.png', dpi=200)

##---------Calculate feature_importance-------
# random forest classifier (after labelling, treat the problem as a supervised learning problem to get feature importance)
randomf = RandomForestClassifier(n_estimators=100,random_state = 42)
randomf.fit(feature_pos, clusters)
# show feature importance
feature_importances_feedback = pd.DataFrame({"feature": feature_pos, "importance": randomf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_feedback.head(20) #20 first important features
#plot feature importance
(pd.Series(randomf.feature_importances_, index=feature_pos.columns)
   .nlargest(5)
   .plot(kind='barh'))

##---------------use training data and train KNN model and save it for future prediction------------
#training features and labels:  feature_pos, train_labels
train_labels=pd.DataFrame(clusters)
filename='knn_train_model_saved.sav'
model_knn = KNeighborsClassifier(n_neighbors=5) # try different k here
model_knn.fit(feature_pos,train_labels.values.ravel())
pickle.dump(model_knn, open(filename, 'wb')) #save the model
#-----------------------------------------------------------------------