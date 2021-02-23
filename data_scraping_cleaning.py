## scraping code

import pandas as pd

!pip install tweepy #intall tweepy in terminal
import tweepy

# create a new account in tweepy api
consumer_key=''
consumer_secret_key=''
access_token=''
access_secret=''

auth=tweepy.OAuthHandler(consumer_key,consumer_secret_key)
auth.set_access_token(access_token,access_secret)
api=tweepy.API(auth,wait_on_rate_limit=True)

# get tweets by hashtag
# get the tweets: q: hashtag word lang:language geocode: lattitude, longitude and radius
# since: time tweet_mode: extended (characters more than 140)
new_tweets=tweepy.Cursor(api.search,\
                         q="#covid OR #covid19 OR \
						    #covid-19 OR #COVID OR #COVID-19 OR #Covid OR \
							#Covid19 OR #Covid-19 OR #COVID19 OR \
							#covid19news OR #Covid19news OR #COVID19news",\
                         lang="en",
                         geocode='39.5,-98,5500km',
                         since='2020-03-01',
                         tweet_mode='extended'
                         #retry_count=5,
                         #retry_delay=5
                         ).\
                         items(100000)
						 
# since there is limit in tweepy api, we can't scrape massive tweets at once
# use backoff strategy to solve this problem
from time import sleep
i=0
allTweets=[]
backoff_counter=1
while True:
    try:
        for tw in new_tweets:
            i=i+1
            inner_list=[tw.user.screen_name,tw.user.location,tw.full_text]
            allTweets.append(inner_list)

        break
    except tweepy.TweepError as e:
        print(i)
        print(e.reason)
        i=0
        sleep(15*backoff_counter)
        backoff_counter=backoff_counter+1
        continue

# save user, location and text as dataframe        
df=pd.DataFrame(data=allTweets,columns=['user','location','text'])
df.to_csv('DATA.csv')

# fuction to strip url
def StripUrl(Str):
    url_pattern=re.compile(r'(?P<url>https?://[^\s]+)')
    url_list=url_pattern.findall(Str)
    for url in url_list:
        Str=Str.replace(url,'')
    return Str
	
#fuction to find hashtag
def findhash(string):
    words=string.split(' ')
    hashtags=[]
    for word in words:
        if '#' in word:
            hashtags.append(word)
    return hashtags
	
# fuction to rehashtag words
def retag(Str,tag):
    for i in puc:
        Str=Str.replace(i,'')
        
    Str=Str.split(' ')
    
    for word in tag:
        if word in Str:
            p=Str.index(word)
            retag='#'+word
            Str[p]=retag
    Str=' '.join(Str)
    return Str
	
	
# fuction to judge whether more than one hashtag in tweets
def htg1(Str):
    if Str.count('#')>=2:
        return 1
    else:
        return 0


# fuction to create hashtag words dataframe	
def dfFreq(li):
    n=len(li)
    tags=[]
    for i in range(0,n):
        tags.append(findhash(li[i]))
    
    alltags=[]
    for tag in tags:
        for j in range(0,len(tag)):
            t=tag[j]
            if ('covid' not in t) and ('Covid' not in t) and ('COVID' not in t)and ('coronavirus' not in t) and ('Coronavirus' not in t) and ('pandemic' not in t)and ('Pandemic' not in t) and ('corona' not in t)and ('Corona' not in t):
                alltags.append(tag[j])
                
    C = Counter(alltags)
    
    vocab_labels, vocab_values = zip(*Counter(C).items())
    sorted_values = sorted(vocab_values)[::-1]
    sorted_labels = [x for (y,x) in sorted(zip(vocab_values,vocab_labels))][::-1]
    df_tagfreq=pd.DataFrame(list(zip(sorted_labels,sorted_values)))
    
    df_tagfreq.columns=['tag','freq']
    
    return df_tagfreq
import string
puc=string.punctuation #punctuation set
	
#convert all letters into lowercase	
df['text']=df['text'].apply(lambda x: x.replace('\n',' '))
df['LowerCaseText']=df['text'].apply(lambda x: x.lower())
# rehashtag words	
tagfreq=dfFreq(df.LowerCaseText)
df['NewText']=df['LowerCaseText'].apply(lambda x: retag(x,tagfreq.word))
	
#sentiment analysis for each tweets
#boto3 comprehend in aws

import boto3
comprehend = boto3.client('comprehend', region_name='us-east-1')
# function to retrieve sentiment 
def neg(text):
    sentiments=comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return sentiments['Sentiment']
	
df['Group']=0	

# in case some tweets can't be sentiment analyzed by boto
# return the error index
# Group here can be negative positive neutral and mixed	
for i in range(0,len(df)):
    try:
        df['Group'][i]=neg(df['text'][i])
        if i%10000==0:
            print(i)
            print('\n')
    except:
        print('error:\n')
        print(i)
        continue
	
## randomly sample 6000 tweets and mannuly label them as rumor or not rumor
## read the data into dataframe
 
df=pd.read_csv('Rumor.csv')


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
	
stopWords = set(stopwords.words('english'))

# fuction to get the combination of the each tweets
# return the list like [1,0,0,0,0,1] for each combination
def get_comb(text,flag,Rumor):
    text=text.split(' ')
    tag=[]
    comb=[]
    if (flag=='NEGATIVE' and Rumor==0):
        f=[1,0,0,0,0,1]
    if (flag=='NEUTRAL'and Rumor==0):
        f=[0,1,0,0,0,1]
    if (flag=='POSITIVE' and Rumor==0):
        f=[0,0,1,0,0,1]
    if (flag=='MIXED' and Rumor==0):
        f=[0,0,0,1,0,1]
        
    if (flag=='NEGATIVE' and Rumor==1):
        f=[1,0,0,0,1,0]
    if (flag=='NEUTRAL'and Rumor==1):
        f=[0,1,0,0,1,0]
    if (flag=='POSITIVE' and Rumor==1):
        f=[0,0,1,0,1,0]
    if (flag=='MIXED' and Rumor==1):
        f=[0,0,0,1,1,0]
    
    for word in text:
        if (('#' in word) and (word.replace('#','') not in stopWords)):
            tag.append(word)
            
    for i in range(0,len(tag)):
        for j in range(0,len(tag)):
            if i<j:
                li=((tag[i],tag[j]),f)
                comb.append(li)
                
    return comb
                                              
df['COMB']=0	
for i in range(0,len(df)):
    df['COMB'][i]=get_comb(df['NewText'][i],df['Group'][i],df['Rumor'][i])	

# get the frequency table of each combination
# save it into dictionary
l=list(df.COMB)
r=[item for sublist in l for item in sublist]

s=set()
newlist={}
for i in range(0,len(r)):
    key=r[i][0]
   # print(key)
    if key not in s:
        s.add(key)
        newlist[key]=r[i][1]
    else:
        newlist[key]=np.array(newlist[key])+np.array(r[i][1])
		
# convert the dictionary into dataframe
DF=pd.DataFrame.from_dict(newlist,orient='index')
	
	
# the vertices in the DF have the cases (a,b) (b,a)
DF.columns=['neg','ne','po','m','rumor','not_rumor']

# check the number of negtive, neutral, positive, mixed, rumor and not_rumor frequency
DF[['neg','ne','po','m','rumor','not_rumor']].sum()
	
# add (a,b) and (b,a) frequency up
keylist=list(newlist.keys())
new_dict={}
keyset=set()
for i in range(0,len(newlist)):
    key=keylist[i]
    temp=(key[1],key[0])
    if ((temp in keylist) and (temp not in keyset)): 
        new_dict[key]=np.array(newlist[key])+np.array(newlist[temp])
        keyset.add(key)
    if temp not in keylist:
        new_dict[key]=np.array(newlist[key])
        
    if i%10000==0:
        print(i)
df_new=pd.DataFrame.from_dict(new_dict,orient='index')
# save the dataframe into csv file
# this is link dataset
df_new.reset_index()
df_new.columns=['vertices','neg','ne','po','m','rumor','not_rumor']
df_new['v1']=0
df_new['v2']=0
for i in range(0,len(df)):
    t=df_new['vertices'][i]
    df_new.v1[i]=t[0]
    df_new.v2[i]=t[1]
    
    if i%10000==0:
        print(i)
df_new.to_csv('fulldfR.csv')

###############################################
# get the list for each hashtag word in tweet
# for instance:#believe [1,0,0,0,0,1]
def get_freq(text,flag,Rumor):
    text=text.split(' ')
    tag=[]
    comb=[]
    if (flag=='NEGATIVE' and Rumor==0):
        f=[1,0,0,0,0,1]
    if (flag=='NEUTRAL'and Rumor==0):
        f=[0,1,0,0,0,1]
    if (flag=='POSITIVE' and Rumor==0):
        f=[0,0,1,0,0,1]
    if (flag=='MIXED' and Rumor==0):
        f=[0,0,0,1,0,1]
        
    if (flag=='NEGATIVE' and Rumor==1):
        f=[1,0,0,0,1,0]
    if (flag=='NEUTRAL'and Rumor==1):
        f=[0,1,0,0,1,0]
    if (flag=='POSITIVE' and Rumor==1):
        f=[0,0,1,0,1,0]
    if (flag=='MIXED' and Rumor==1):
        f=[0,0,0,1,1,0]
    
    for word in text:
        if (('#' in word) and (word.replace('#','') not in stopWords)):
            tag.append(word)
            
    for i in range(0,len(tag)):
        li=(tag[i],f)
        comb.append(li)
                
    return comb
               
df['COMB']=0
for i in range(0,len(df)):
    df['COMB'][i]=get_freq(df['NewText'][i],df['Group'][i],df['Rumor'][i])

l=list(df.COMB)
r=[item for sublist in l for item in sublist]

# to get the words frequency
s=set()
newlist={}
for i in range(0,len(r)):
    key=r[i][0]
   # print(key)
    if key not in newlist.keys():
        newlist[key]=np.array(r[i][1])
    else:
        newlist[key]=np.array(newlist[key])+np.array(r[i][1])
DF=pd.DataFrame.from_dict(newlist,orient='index')
DF.columns=['neg','ne','po','m','rumor','not_rumor']
DF.to_csv('nodeInfo.csv')