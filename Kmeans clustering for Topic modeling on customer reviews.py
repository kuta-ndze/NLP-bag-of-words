#!/usr/bin/env python
# coding: utf-8

# # ToC
# 
# - [1 - Introduction](#1)
# - [2 - Understanding Nltk](#2)
# - [3 - Importing Libraries](#3)
# - [4 - Importing Dataset](#4)
# - [5 - About dataset](#5)
# - [6 - Cleaning data](#6)
# - [7 - Tokenizing](#7)
# - [8 - Vectorizer](#8)
# - [9 - Clustering](#9)
#    - [9.1 - Wordcloud](#9.1)
# - [10 - Generic function homogeinity](#10)

# <a name='1'></a>
# #### Introduction to topic modeling
# * Large amounts of data are collected everyday. As more information becomes available, it becomes difficult to access what we are looking for. So , we need tools and techniques to organize, search and understand vast quantities of information. Topic modeling provide us with methods to organize, understand and summarize large collections of textual information. 
# * How do we identify topics in a given text document from a large amounts of data with lots of information. We should be able to organize,  search , understand the  information in the data. Topic modeling helps in :
#   * Discovering the hidden topical patterns that are present in the collection
#   * Annotating documents according to these topics the data so that it can be utilized based on the topic that is identified and certain steps can be taken based on the business directions
#   * Using these annotations to organize, search and summarize texts.
# * Topic modeling can also be describe as a method of finding a group of words from a collection of documents that best represent the information in the collection. Also as a form of text mining and a way of obtaining recurring pattern of words in  the textual material.
# * In this project we use the Unsupervised machine learning techniques which will help in clustering/grouping the data/reviews to identify the main topics or idea in the sea of text . This is just some  corpus of data with no labels associated with it.
# * We will focused on Twitter data which tend to be more complex and Noisy  compared to data obtained from review forms or any other textual information because in the twitter data not only people have their own colloquial languages but also a lot of noise present.
# * We will learn how to clean the noise then use that data to cluster and identify the main topics that people are talking about then depending on what ever is the business decision they can take certain steps about it. In order to do so we will use the NLTK package.

# <a name='2'></a>
# #### Understanding Nltk
# 
# * Natural Language Toolkit provides tools to enable computers understand natural language. leading platform for building python program to work with human language data, very easy to deal with computer languages as they tend to be very structured data. however the human languages is a very unstructured data i.e same thing can be said in a variety of ways and it mean the same thing . In a computer they are different sentences which can mean different things so nltk helps to cleaning and preprocessing  the data in such a way to make it a little more structured from its own unstructured form. It provide quite easy to use interfaces and provide suits for text processing libraries for things like 
#   * classification, 
#   * Tokenization(separating out the words and moving punctuations) converting to bag of words, 
#   * Stemming which gets to the core same word can have alot of prefixes and suffixes but at the core mean the same thing .
#   * Tagging each word can be tag 
#   
# * It is free and also open source, and community driven project. you can directly import to use nltk however you will initial have to install nltk and given that it has a vast tool and usually takes a very long time to install all of it. can take up to an hour. instead we can install part of it.

# In[ ]:


#nltk.download_gui()


# <a name='3'></a>
# #### Importing Libraries
# 

# In[61]:


import nltk 
import numpy as np 
import pandas as pd 
import re #remove regex
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Get multiple output in the same cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ignore all warnings

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# In[5]:


# Display all rows and columns of a dataframe instaead of a truncated version

from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# <a name='4'></a>
# #### Importing and exploring dataset

# In[6]:


raw_data = pd.read_csv("tweets.csv", encoding= 'ISO-8859-1')
print(len(raw_data))


# In[7]:


df = raw_data
df 


# <a name='5'></a>
# #### About Data
# 
# * Twitter data with lots of Noise on reviews 21047 tweets with 4 attributes username, date , tweet and mention i.e a data about vodafone which is a telecome company in India.

# In[8]:


#check for duplicate tweets
unique_text = df.tweet.unique()
print(len(unique_text))


# In[9]:


df.head()


# In[13]:


df['tweet'][444]


# * Given this its almost  impossible for a person to go through all the tweet to be able to identify which area requires the most addressing, it 
# it the services, network, number porting unless someone manually goes through all the tweets present 

# <a name='6'></a>
# #### Cleaning the data

# In[14]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    #print(input_txt)
    #print(r)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    
    return input_txt


# In[18]:


# Remove @ mentions np.vectorize is another way of writing a for loop to loop the tweets data in the dataframe
df['Clean_text'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
df['Clean_text']


# In[19]:


#removing punctuations or non letters with space
df['Clean_text'] = df['Clean_text'].str.replace("[^a-zA-Z#]", " ")
df['Clean_text']


# In[21]:


#convert strings to lower case this makes sure computer understand that Please in capital letters and small mean same thing
# as humans will understand.
df['Clean_text'] = df['Clean_text'].str.lower()
df['Clean_text']


# In[22]:


# collapsing text/remove space and removing words with lenght less than 2 .split() splits string to a list hwere each wor is a list item
df['Clean_text']= df['Clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
df.head() 


# <a name='7'></a>
# #### Tokenizing and identifying special words

# In[23]:


# basically splitting each word, it also makes sure if they are any full stop at end of word gets romoved.
tokenized_tweet = df['Clean_text'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[24]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
df['Clean_text'] = tokenized_tweet


# In[25]:


df 


# In[26]:


df.loc[:, ('Clean_text')]


# In[27]:


# drop duplicates
df.drop_duplicates(subset=['Clean_text'], keep ='first', inplace= True)


# In[29]:


df.reset_index(drop=True, inplace=True)


# In[30]:


df


# In[31]:


df['Clean_text_length'] = df['Clean_text'].apply(len)
df.head() 


# In[32]:


df[df['Clean_text_length'] == 0]


# In[34]:


# just to make sure this is not an artifact of our previous preprocessing
raw_data[raw_data['username'] == 'omanmessi']


# In[35]:


# looks like these are tweets with differen languages or just hashtags
df[df['Clean_text_length']==0]['Clean_text']
#we can simply drop these tweets
indexes =df[df['Clean_text_length']==0]['Clean_text'].index
indexes 


# In[36]:


df.drop(index = indexes, inplace= True)


# In[37]:


df.info()


# In[38]:


df.reset_index(drop= True, inplace= True)
df.info() 


# In[39]:


df['Clean_text'].head() 


# <a name='8'></a>
# #### Vectorizer 
# converts a collection of raw documents to a matrix of TF-IDF features. two types of vectorizers 
#   * TfidVectorizer: In information retrieval, tf-idf or TFIDF, short for term frequency - inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.This is use when dealing with NLP on bigger textual informations with has paragraphs in this case is better to use
#   * CountVectorizer: Converts a collection of text documents to a matrix of token counts. The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary . Number of features will be equal to the vocabulary size found by analyzing the data.

# In[40]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word', ngram_range=(1,1), stop_words='english', min_df=0.0001, max_df=0.7)
count_vect.fit(df['Clean_text'])
desc_matrix = count_vect.transform(df['Clean_text'])
desc_matrix


# In[42]:


desc_matrix.toarray() 


# In[43]:


desc_matrix.shape 
#6743 features is the vocabulary of the tweets which are present.


# <a name='9'></a>
# #### Clustering

# In[44]:


get_ipython().system('pip3 install KMeans')
get_ipython().system('pip3 install wordcloud')


# In[45]:


from sklearn.cluster import KMeans 
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests 


# In[46]:


num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(desc_matrix)
clusters = km.labels_.tolist()


# In[47]:


tweets = {'Tweet' : df['Clean_text'].tolist(), 'Cluster': clusters}
frame = pd.DataFrame(tweets, index= [clusters])
frame 


# In[48]:


frame['Cluster'].value_counts()


# In[49]:


cluster_0 = frame[frame['Cluster'] == 0]
cluster_0 


# In[58]:


#Mask is backdrop in which your image of words will be plotted
def wordcloud(cluster):
  # combining the image with the dataset
  Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

  # We use the ImageColorGenerator library from Wordcloud 
  # Here we take the color of the image and impose it over our wordcloud
  image_colors = ImageColorGenerator(Mask)

  # Now we use the WordCloud function from the wordcloud library 
  wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(cluster)

  # Size of the image generated 
  plt.figure(figsize=(10,20))

  # Here we recolor the words from the dataset to the image's color
  # recolor just recolors the default colors to the image's blue color
  # interpolation is used to smooth the image generated 
  plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

  plt.axis('off')
  plt.show()


# In[59]:


#create a very long paragraph
cluster_0_words = ''.join(text for text in cluster_0['Tweet'])


# <a name='9.1'></a>
# ####  word cloud

# In[62]:


# this tells us about the problem related to services !! Different clusters are form each time

wordcloud(cluster_0_words)


# In[64]:


cluster_1 = frame[frame['Cluster'] == 1]


# In[65]:


cluster_1 


# In[66]:


cluster_1_words = ''.join(text for text in cluster_1['Tweet']) 


# In[67]:


wordcloud(cluster_1_words)


# In[69]:


# Clustering with 8 clusters  again in this case there is no scientific way to choose the clusters number, it cames with experience.
# depends on the output you see in the clusters 8 could not be the good choice

num_clusters = 8
km = KMeans(n_clusters=num_clusters)
km.fit(desc_matrix)
clusters = km.labels_.tolist()


# In[70]:


tweets = {'Tweet' : df['Clean_text'].tolist(), 'Cluster': clusters}
frame = pd.DataFrame(tweets, index= [clusters])
frame 


# In[71]:


frame['Cluster'].value_counts()


# In[72]:


cluster_0 = frame[frame['Cluster'] == 0]
cluster_0 


# In[73]:


cluster_0 = frame[frame['Cluster'] == 0]
cluster_0_words = ' '.join(text for text in cluster_0['Tweet']) 
wordcloud(cluster_0_words)


# In[74]:


cluster_1 = frame[frame['Cluster'] == 1]
cluster_1_words = ' '.join(text for text in cluster_1['Tweet']) 
wordcloud(cluster_1_words)


# In[76]:


cluster_2 = frame[frame['Cluster'] == 2]
cluster_2_words = ' '.join(text for text in cluster_2['Tweet']) 
wordcloud(cluster_2_words)


# In[77]:


cluster_3 = frame[frame['Cluster'] == 3]
cluster_3_words = ' '.join(text for text in cluster_3['Tweet']) 
wordcloud(cluster_3_words)


# In[78]:


cluster_4 = frame[frame['Cluster'] == 4]
cluster_4_words = ' '.join(text for text in cluster_4['Tweet']) 
wordcloud(cluster_4_words)


# In[79]:


cluster_5 = frame[frame['Cluster'] == 5]
cluster_5_words = ' '.join(text for text in cluster_5['Tweet']) 
wordcloud(cluster_5_words) 


# In[80]:


cluster_6 = frame[frame['Cluster'] == 6]
cluster_6_words = ' '.join(text for text in cluster_6['Tweet']) 
wordcloud(cluster_6_words) 


# In[81]:


cluster_7 = frame[frame['Cluster'] == 7]
cluster_7_words = ' '.join(text for text in cluster_7['Tweet']) 
wordcloud(cluster_7_words)


# In[83]:


# so here we see that 8 clusters is alot , however we have a ball mark to choose number of clusters between 2-8
# save the clustered tweets

frame.to_csv('clustered_tweets.csv')  


# <a name='10'></a>
# #### Generic function homogeinity

# In[91]:


# lets define a dunction to identify the topics we want and iterate through clusters

def identify_topics(df, desc_matrix, num_clusters):
    km =KMeans(n_clusters =num_clusters)
    km.fit(desc_matrix)
    clusters = km.labels_.tolist()
    tweets = {'Tweet' : df['Clean_text'].tolist(), 'Cluster': clusters}
    frame = pd.DataFrame(tweets, index = [clusters])
    print(frame['Cluster'].value_counts())
    
    for cluster in range(num_clusters):
        cluster_words = ' '.join(text for text in frame[frame['Cluster'] == cluster]['Tweet'])
        wordcloud(cluster_words) 
    


# In[93]:


identify_topics(df, desc_matrix, 6)
# 6 appears to be the ideal number of clusters i.e looking for homogeinity in the clusters not having too many diff topics in it.
# Or topic divided into many clusters
# IF this goes as an input in the business the business decision makers will know which area to allocate more budget or make more efforts towards where 
# The customers are facing much problems.

