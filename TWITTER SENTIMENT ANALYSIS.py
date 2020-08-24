import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

df=pd.read_csv("train.csv",encoding="ISO-8859-1")

df.isnull().sum()

y=df["Sentiment"]

message=df["SentimentText"]

ps=PorterStemmer()

#--------------------------------------removing @username from tweet----------------------------------
message=message.str.replace("@[\w]*","")

#------------------------------------------REMOVING HYPERLINK FROM TWEET-----------------------------
message = message.str.replace('https?:\/\/\S+', '')

#----------------------------------------------removing RT from tweet--------------------------------
message=message.str.replace('RT[\s]+', '')


#---------------------------------------removing unwanted symbols and stopwords-----------------------
corpus=[]
for i in range(len(message)):
    review=re.sub("[^a-zA-Z]"," ",message[i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(j) for j in review if not j in stopwords.words("english")]
    review=" ".join(review)
    corpus.append(review)

#-------------------understanding the common word used in wordcloud------------------------------------

all_words=" ".join([i for i in corpus])

from wordcloud import WordCloud

wc=WordCloud(width=1200,height=400,random_state=20,max_font_size=120).generate(all_words)
plt.figure(figsize=(30,20))
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.show()

#-----------------------------word matrix for 0 label tweet --------------------------------------------

tweet_cl=pd.DataFrame(corpus,columns=["clean_tweet"])

df_new=pd.concat([df,tweet_cl],axis=1)

words_0=" ".join([i for i in df_new.loc[df_new["Sentiment"]==0,"clean_tweet"]])


wc_0=WordCloud(width=1200,height=400,random_state=20,max_font_size=120).generate(words_0)
plt.figure(figsize=(30,20))
plt.imshow(wc_0,interpolation="bilinear")
plt.axis("off")
plt.show()



#---------------------------------word matrix for 1 label tweet--------------------------------------

words_1=" ".join([i for i in df_new.loc[df_new["Sentiment"]==1,"clean_tweet"]])


wc_1=WordCloud(width=1200,height=400,random_state=20,max_font_size=120).generate(words_1)
plt.figure(figsize=(30,20))
plt.imshow(wc_1,interpolation="bilinear")
plt.axis("off")
plt.show()

#--------------------------------------------------------------------------------------------------------

words = nltk.word_tokenize(words_1)


b=nltk.FreqDist(words)

d=pd.DataFrame({"words":list(b.keys()),"count":list(b.values())})

most_frequent=d.loc[d["count"]>=1000]

sns.set(rc={"figure.figsize":(15,10)})
sns.barplot(x=most_frequent["words"],y=most_frequent["count"])


#----------------------------SENTIMENT ANALYSIS---------------------------------------------------------------------------

df_new=df_new.drop(["ItemID","SentimentText"],axis=1)

from textblob import TextBlob

def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
   return  TextBlob(text).sentiment.polarity


df_new['Subjectivity'] = df_new["clean_tweet"].apply(getSubjectivity)
df_new['Polarity'] = df_new["clean_tweet"].apply(getPolarity)


def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'
    
df_new['Analysis'] = df_new['Polarity'].apply(getAnalysis)

plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
sns.countplot(df_new["Analysis"])
plt.show()











