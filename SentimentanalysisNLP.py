#!/usr/bin/env python
# coding: utf-8

# ## Natural Language Processing with Bag-of-words

# ### Importing the libraries

# In[1]:


#reviews and dependent variable are seperated by tabs.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Importing the dataset 

# In[3]:


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)  #delimiter alias for sep


# In[4]:


print(dataset.head())
print(dataset.shape)


# ### Cleaning text

# In[16]:


#punctuation, capital letters, lower case, verbes constructed differently they should be undastandable
#import tools/library to clean text 
#regular expression use simplify the reviews anything thats not a letter ^ means not
import re                  
import nltk 
#allows us to download the end symbol of stopwords that are not relevant whether review is positive or negative, we will remove them like 'the', 'an, we, apple, etc'
nltk.download('stopwords')
from nltk.corpus import stopwords
#Steming consist of taking only the root of a word that indicate enough about what the word mean e.g loved to love
from nltk.stem.porter import PorterStemmer    
corpus = [] #hold the clean reviews in this list
for i in range(0, 1000):
    #replace any character which is not  a-z or A-Z with a space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  
    #convert reviews to lower case
    review = review.lower()
    #split review into its different words
    review = review.split()
    #create sparse matrix with each column representing a word using stemming
    ps = PorterStemmer()
    #we came back to fix the not that was not included
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')  #will not include the not from the stopwords
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]  #removing stopwords in review with a for row
    review = ' '.join(review) #join each word from steming to a sting but they should be a space ' ' 
    corpus.append(review)


# In[17]:


print(corpus)


# * After comparing the corpus output, they are important words left out by the stopwords like crust good instead of crust not good so we go back to fix this

# ### Creating bag of words model

# In[22]:


#tokenization process with scikit learn after creating sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #put all words into the columns
Y = dataset.iloc[:,-1].values


# In[23]:


#number of columns in X
#1566 words resulting from tokenization with 1 to columns having words in review and 0 not in review
#to get the 1500 most frequent words with put parameter in countvectorizer
len(X[0])


# ### Splitting data into training and test set 

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# ### Training the Naive Bayes model on the Training set 

# In[26]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# ### Predicting new results 

# In[29]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# ### Confusion Matrix 

# In[30]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# * Model could be improved on.
