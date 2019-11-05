# -*- coding: utf-8 -*-
''' This .py file contains functions used for visualization'''

def plot_two_Histograms(data1,data2,title_txt,file_name):
    ''' function to plot histogram of features for 2 clusters '''
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    #create legend
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['#0504aa','salmon']]
    labelss= ["Vague Feedback","Specific Feedback"]
    
    fig, ax = plt.subplots(figsize=(10,5))
    bins=np.histogram(np.hstack((data1,data2)), bins=50)[1]
    n, bins, patches = ax.hist(x=data1, bins=bins, color='#0504aa',alpha=0.7, rwidth=0.85)
    n, bins, patches = ax.hist(x=data2, bins=bins, color='salmon',alpha=0.7, rwidth=0.85)                          
    plt.grid(axis='y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title_txt)
    plt.text(23, 45, r'$\mu=15, b=3$')
    plt.legend(handles, labelss)
    plt.savefig(file_name)
#----------------------------------------------------       
def plot_clusters_pca(principalDf,labels): 
    ''' function to plot clusters using PCA '''
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    colors = ['r', 'g']
    targets=[0,1]

    for target, color in zip(targets,colors):
        indicesToKeep = labels== target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
        ax.legend(targets)
        ax.grid()
#----------------------------------------------------        
def plot_explained_variances(exp_variance,cum_exp_variance):        
    '''function to plot explained variances'''
    import matplotlib.pyplot as plt
    
    plt.bar(range(1,10), exp_variance, alpha=0.5,align='center', label='individual explained variance')
    plt.step(range(1,10), cum_exp_variance, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.show()                
#----------------------------------------------------
def plot_hist(data,title_txt):
    '''function to plot histogram of features'''
    import matplotlib.pyplot as plt
    
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title_txt)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
#----------------------------------------------------    
def show_wordcloud(data, title = None):
    '''function to plot wordcloud'''
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
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
def plot_word_frequency(word_rank,word_freq,word_names):
    '''function to plot frequency of word occurence'''
    import matplotlib.pyplot as plt
    
    plt.title("Word Frequencies")
    plt.ylabel("Log of Total Number of Occurrences")
    plt.xlabel("Log of Rank of word")
    plt.loglog(word_rank,word_freq,basex=10,label=word_names)
    plt.show()
#----------------------------------------------------    
def plot_nonlog_word_frequency(word_rank,word_freq):
    '''function to plot non-logarithm scatter plot of term frequencies'''
    import matplotlib.pyplot as plt

    plt.title("Word Frequencies")
    plt.ylabel("Total Number of Occurrences")
    plt.xlabel("Rank of word")
    plt.scatter(word_rank,word_freq);
    plt.show()   
#----------------------------------------------------
def draw_sentiment_barplot(count_values,count_id):
    '''function to draw sentiment bar plot'''
    import matplotlib.pyplot as plt
    
    plt.bar(count_values, count_id)
    plt.xlabel('Feedback Sentiment')
    plt.ylabel('Number of Feedback')
    plt.show()   
#----------------------------------------------------    
def plot_sentimentscore_frequency(sentiment_df):
    '''function to plot sentiment distribution for positive, negative, and neutral feedback'''
    import seaborn as sns
    
    for sentiment in ['negative','positive','neutral']:
        subset = sentiment_df[sentiment_df['sentiment_score'] == sentiment]   
    # Draw the density plot
        if sentiment == 'negative':
            label_plot = "Bad Feedback"
        elif sentiment == 'positive':
            label_plot = "Good Feedback"
        else:   
            label_plot = "Neutral Feedback"
        sns.distplot(subset['sentiments_compound'], hist = False, label = label_plot) 
#----------------------------------------------------       
def sentiment_plot(values, names):
	'''function to plot horizontal sentiment scores for any feedback '''
    import matplotlib.pyplot as plt
    	
    values[0]=values[0]*100
    values[1]=values[1]*100
    values[2]=values[2]*100
    				
    values_incr = [None] * 3
    values_incr[0]=values[0]
    values_incr[1]=values[1]+values[0]
    values_incr[2]=values[2]+values[1]+values[0]
    					
    label="Feedback Sentiment "
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(values)))
    		
    fig, ax = plt.subplots(figsize=(9.2, 1))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)

    for i, (colname, color) in enumerate(zip(names, category_colors)):
        widths = values[i]
        starts = values_incr[i] - widths
        ax.barh(label, widths, left=starts, height=0.5,label=colname, color=color)
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        #for y, (x, c) in enumerate((xcenters, widths)):
        ax.text(xcenters, 0, str(int(round(widths,0)))+ '%', ha='center', va='center',color=text_color) #add percentage format to this
    ax.legend(ncol=len(names), bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')
    plt.savefig('sentiment_scores_plot.png', dpi=200, bbox_inches='tight')
		
    return fig   
        