<p aligne = "center">
<a href="https://kuta-ndze.github.io/" target="_blank" rel="noopener noreferrer"><img alt="Eample Portfolio URL" src="https://img.shields.io/badge/Portfolio-%23000000.svg?style=for-the-badge&logo=firefox&logoColor=#FF7139" height="25"></a> 
<a href="https://github.com/kuta-ndze"><img alt="github URL" src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" height="25"></a>
<a href="mailto:kutaceldrick880@gmail.com"><img alt="Mailto" src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" height="25"></a>
<a href="https://www.linkedin.com/in/kuta-n-celdrick-b808ba169/" target="_blank" rel="noopener noreferrer"><img alt="Linkedin URL" src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" height="25">
<a href="https://twitter.com/kutandze" target="_blank" rel="noopener noreferrer"><img alt="Twitter URL" src="https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white" height="25"></a></p><br>

**This Repo Contains two personal projects outlined below and all files attached in the repo**

## `1. Sentiment Analysis`

- An example intuition to bag of words model in NLP using Kirill Eremenko Restaurant reviews intuition dataset.
- You could improve more on the model could be improved further to still be able to get the intuition behind it.
  - [**sentimentalanalysisNLP.py**](https://github.com/kuta-ndze/NLP-bag-of-words/blob/main/SentimentanalysisNLP.py)

## `2. Topic Modelling`

**Objective**

- In this project, we want to group customers reviews on twitter corpus based on recurring
  patterns. We should be able to get a sense of the specific topic in each cluster, what the customers are complaining about
  based on specific patterns. The twitter corpus contains a lot of noise and we will try to minimize this and create sense out of the data.

**Data**

- The data used is Twitter data with lots of Noise on reviews. 21047 tweets with 4 attributes username, date , tweet and mention i.e a data about vodafone which is a telecom company [tweets.csv](https://github.com/kuta-ndze/Natural_Language_Processing/blob/main/tweets.csv).

**Methodology**

- The ML technique used in this project is the kmeans clustering which is an unsupervised model to be able to extract some patterns.

1. `Data Cleaning with Pattern Removal`
   - Removing mentions with @
   - Replacing non-alphabets with empty space
   - Convert Capital cases to lower cases for computer comprehension
   - Collapse all spaces and remove words with lengths less than 2
2. `Tokenizing data and Identify Special Instances of Tweets`
   - Create a list for each row of the clean text by making each word a standalone this also takes care of any full stops at end of text removes.
   -
