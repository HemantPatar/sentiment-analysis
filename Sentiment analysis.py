#!/usr/bin/env python
# coding: utf-8

# Story sentiment analysis--
# The short story was taken from below website & the short story is "the night came slowly" 
# https://americanliterature.com/author/kate-chopin/short-story/the-night-came-slowly
# The story is saved as text file for analysis.

# In[4]:


import string
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')
#nltk.download('vader_lexicon')


# In[5]:


#Loading text file & converting it into lower case.
text = open('story.txt',encoding="utf-8").read()

lower_case = text.lower()


# In[6]:


# str.maketrans removes any punctuations 

cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))


# In[7]:


# Using word_tokenize to tokenize sentence into words

tokenized_words = word_tokenize(cleaned_text, "english")


# In[8]:


# Removing Stop Words
final_words = []

for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)


# In[9]:


# Lemmatization - From plural to single + Base form of a word (example better-> good)

lemma_words = []

for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)


# In[10]:


emotion_list = []

with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in lemma_words:
            emotion_list.append(emotion)
            
print("People emotions from the text \n", emotion_list, '\n \n')


w = Counter(emotion_list)
print("Count of each emotion \n", w)


# In[11]:


# Now we use SentimentIntensityAnalyzer on processed text
sia = SentimentIntensityAnalyzer()

sent = cleaned_text 

print (sia.polarity_scores(sent))


# In[14]:


def sentiment_analyse(sentiment_text):
    
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    
    if score['neg'] > score['pos']:
        print("\n     ******Negative Sentiment*******")
        
    elif score['neg'] < score['pos']:
        print("\n     ******Positive Sentiment*******")
        
    else:
        print("Neutral Sentiment")
        
sentiment_analyse(cleaned_text)

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
# plt.savefig('graph.png')
plt.show()

