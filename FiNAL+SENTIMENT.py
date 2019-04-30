
# coding: utf-8

# In[20]:


import os 
os.chdir('C:\\Users\\Dell\\Documents\\Sentiment')


# In[21]:


os.getcwd()


# In[22]:


import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
pd.set_option('display.max_colwidth', 200)


# In[23]:


Train=pd.read_csv('train.csv')


# In[24]:


Train.head()


# In[25]:


Test=pd.read_csv('test.csv')


# In[26]:


Test.head()


# In[27]:


Train['label'].value_counts(normalize=True)


# In[28]:


Train.head()


# In[29]:


Train['clean_tweet'] = Train['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))


# In[30]:


Test['clean_tweet'] = Test['tweet'].apply(lambda x: re.sub(r'https\S+', '', x))


# In[31]:


Train['clean_tweet']


# In[32]:


punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
Train['clean_tweet'] = Train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
Test['clean_tweet'] = Test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))


# In[33]:


Train['clean_tweet']


# In[34]:


Train['clean_tweet'] = Train['clean_tweet'].str.lower()
Test['clean_tweet'] = Test['clean_tweet'].str.lower()


# In[35]:


Train['clean_tweet'] = Train['clean_tweet'].str.replace("[0-9]"," ")
Test['clean_tweet'] = Test['clean_tweet'].str.replace("[0-9]"," ")


# In[36]:


Train['clean_tweet']


# In[37]:


Train['clean_tweet'] = Train['clean_tweet'].apply(lambda x:' '.join(x.split()))
Test['clean_tweet'] = Test['clean_tweet'].apply(lambda x: ' '.join(x.split()))


# In[38]:


Train['clean_tweet']


# In[40]:


# import spaCy's language model
nlp = spacy.load('en_core_web_sm')
disable=(['parser', 'ner'])
#nlp = spacy.load('en_core_web
# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output


# In[42]:


Train['clean_tweet'] = lemmatization(Train['clean_tweet'])
Test['clean_tweet'] = lemmatization(Test['clean_tweet'])


# In[43]:


Train.sample(10)

