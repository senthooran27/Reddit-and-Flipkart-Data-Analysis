#!/usr/bin/env python
# coding: utf-8

# In[6]:


import praw
reddit = praw.Reddit(client_id ='tbxZ6eKXACO2JNFKHCQjPg', client_secret='_ZhaSATEothB1c2saebqKZRj7_kKkQ', user_agent='sen')


# In[7]:


import pandas as pd
posts = []
topic = reddit.subreddit('MachineLearning')
for post in topic.hot(limit=10):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
print(posts)


# In[8]:


hot_posts = reddit.subreddit('all').hot(limit=10)
for post in hot_posts:
    print(post.title)


# In[9]:


url= "https://www.reddit.com/r/CryptoCurrency/comments/16n6h6v/daily_crypto_discussion_september_20_2023_gmt0/"
submission = reddit.submission(url=url)
post_comments = []
submission.comments.replace_more(limit=100)
for comment in submission.comments.list():
    post_comments.append(comment.body)
post_comments = post_comments[1:]
post_comments


# In[10]:


df = pd.DataFrame(post_comments)
df1 = pd.DataFrame(post_comments)
df.to_csv('jawaan.csv',index = False)

from textblob import TextBlob
import numpy as np
df['Sentiment'] = np.nan
df.columns = ['Comment','Sentiment']
df


# In[11]:


for index, row in df.iterrows():
    comment_polarity = TextBlob(row['Comment']).sentiment.polarity
    df.at[index,'Sentiment'] = comment_polarity
df


# In[12]:


unwanted_index = []
for index,row in df.iterrows():
    if row['Sentiment'] == 0.0:
        unwanted_index.append(index)
len(unwanted_index)


# In[13]:


final_data = df.drop(unwanted_index)
final_data


# In[14]:


final_data.describe()


# In[15]:


positive_count = 0
negative_count = 0
for index, row in final_data.iterrows():
    if row['Sentiment'] > 0:
        positive_count += 1
    elif row['Sentiment'] < 0:
        negative_count += 1
print(positive_count, negative_count)


# In[16]:


import seaborn as sns
sns.displot(x=final_data['Sentiment'])


# In[17]:


import spacy
nlp = spacy.load("en_core_web_sm")
complete_doc = nlp(str(post_comments))
words = [token.text for token in complete_doc if not token.is_stop and not token.is_punct and token.pos_ == 'PROPN']
from collections import Counter
word_freq = Counter(words)
common_words = word_freq.most_common(10)
common_words


# In[18]:


positive_words = [token.text for token in complete_doc if not token.is_stop and not token.is_punct and TextBlob(str(token)).sentiment.polarity > 0]
positive_freq = Counter(positive_words)
common_positive_words = positive_freq.most_common(10)
common_positive_words
a,b = zip(*common_positive_words)
a = np.array(a)
b = np.array(b)
sns.barplot(x=a,y=b,color = 'blue').set_title('Top 10 frequently used positive words')


# In[19]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk
nltk.download('wordnet')
stemmer = PorterStemmer()
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
processed_docs = df1[0].map(preprocess)
processed_docs[:10]


# In[20]:


#Bag of Words
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]






bow_doc_310 = bow_corpus[310]
for i in range(5):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_310[i][0],dictionary[bow_doc_310[i][0]],bow_doc_310[i][1]))


# In[21]:


from gensim import corpora, models
ldamodel = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
topics = ldamodel.print_topics(num_words=8)
for topic in topics:
   print(topic)


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
x = vectorizer.fit_transform(df1[0])
#n_Components  = Number of Topics
lsa = TruncatedSVD(n_components=10,n_iter=100)
lsa.fit(x)
terms = vectorizer.get_feature_names_out() 
print(lsa.components_)


# In[23]:


for ind,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedTerms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:7]
    print("Concept %d" % ind)
    for term in sortedTerms:
        print(term[0])
    print(" ")


# In[24]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
data = pd.read_csv("C:/Users/senth/Downloads/flipkart_data.csv")
data.head()


# In[25]:


pd.unique(data['rating'])


# In[26]:


sns.countplot(data=data,
              x='rating',
              order=data.rating.value_counts().index)


# In[27]:


from tqdm import tqdm


def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in tqdm(text_data):
        # Removing punctuations
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Converting lowercase and removing stopwords
        preprocessed_text.append(' '.join(token.lower()
                                        for token in nltk.word_tokenize(sentence)
                                        if token.lower() not in stopwords.words('english')))

    return preprocessed_text
# rating label(final)
pos_neg = []
for i in range(len(data['rating'])):
    if data['rating'][i] >= 5:
        pos_neg.append(1)
    else:
        pos_neg.append(0)

data['label'] = pos_neg


# In[5]:


preprocessed_review = preprocess_text(data['review'].values)
data['review'] = preprocessed_review


# In[6]:


data.head()


# In[7]:


data["label"].value_counts()


# In[8]:


consolidated = ' '.join(
    word for word in data['review'][data['label']==1].astype(str))
wordCloud = WordCloud(width=1600, height=800,
                      random_state=21, max_font_size=110)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:




