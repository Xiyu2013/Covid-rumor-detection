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


##########################model##########################
###model data prepration
dflink=pd.read_csv('fulldfR.csv')
dflink.rename(columns={'Unnamed: 0':'vertices'},inplace=True)
dfnode=pd.read_csv('nodeInfo.csv')
dflink.drop(['Unnamed: 0','vertices'],axis=1,inplace=True)

dfnode.rename(columns={'Unnamed: 0':'tag'},inplace=True)
dfnode.columns=['tag','neg_n','ne_n','p_n','m_n','r','not_r']

# keep the tag words both in dflink and dfnode
dflink=dflink.merge(dfnode,how='inner',left_on='v1',right_on='tag')
dflink.drop(['tag','neg_n','ne_n','p_n','m_n','r','not_r'],axis=1,inplace=True)
dflink=dflink.merge(dfnode,how='inner',left_on='v2',right_on='tag')
dflink.drop(['tag','neg_n','ne_n','p_n','m_n','r','not_r'],axis=1,inplace=True)

#create key for each hashtag word
l=list(dflink.v1)+list(dflink.v2)
s=set(l)
l=list(s)

dfkey=pd.DataFrame()
dfkey['tag']=l
dfkey['key']=[i for i in range(0,len(dfkey))]

dfnode=dfnode.merge(dfkey,how='inner',left_on='tag',right_on='tag')
# label each link as rumor or not_rumor
temp=[]
for i in range(0,len(dflink)):
    a=dflink['rumor'][i]
    b=dflink['not_rumor'][i]
    c=a/(a+b)
    temp.append(c)
dflink['rate']=temp
dflink['label']=dflink['rate'].apply(lambda x: 1 if x>dflink['rate'].mean() else 0)

# label each node as rumor and not_rumor
temp=[]
for i in range(0,len(dflink)):
    a=dfnode['rumor'][i]
    b=dfnode['not_rumor'][i]
    c=a/(a+b)
    temp.append(c)
dfnode['rate']=temp
dfnode['label']=dfnode['rate'].apply(lambda x: 1 if x>dfnode['rate'].mean() else 0)

# torchtext for glove 
#dgl for GCN model
!pip install torchtext
!pip install dgl
import torch
import torchtext
import dgl
import numpy as np
import matplotlib.pyplot as plt

## import nesassary package for model
import dgl.nn as dglnn 
import torch.nn as nn 
import torch.nn.functional as F 
from dgl.data import * 

# get the embeddings for each word
glove=torchtext.vocab.GloVe(name='twitter.27B',dim=100)
dfnode['word']=dfnode['tag'].apply(lambda x:x.replace('#',''))

l=[]
for word in dfnode.word:
    l.append(glove[word].tolist())

# create tensor for the GCN model
# bidirected model
t=torch.tensor(l)
src=torch.tensor(list(dflink.key1))
dst=torch.tensor(list(dflink.key2))
train=[]
validation=[]
test=[]
for i in range(0,2977):
    if i<1500 or i>=2877:
        train.append(True)
        validation.append(False)
        test.append(False)
    if i>=1500 and i<2000:
        train.append(False)
        validation.append(True)
        test.append(False)
    if i>=2000 and i<2877:
        train.append(False)
        validation.append(False)
        test.append(True)
		
# undirected graph
g=dgl.graph((src,dst))
bg=dgl.to_bidirected(g)
bg=dgl.remove_self_loop(bg)
bg.ndata['feat']=t
label=dfnode['label']
bg.ndata['label']=torch.tensor(label)
bg.ndata['train_mask']=torch.tensor(train)
bg.ndata['val_mask']=torch.tensor(validation)
bg.ndata['test_mask']=torch.tensor(test)
node_features=bg.ndata['feat']
node_labels=bg.ndata['label']
train_mask=bg.ndata['train_mask']
valid_mask=bg.ndata['val_mask']
test_mask=bg.ndata['test_mask']
n_features=node_features.shape[1]
n_labels=2

		
		
		
# two layer GCN model		
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.2):
        super().__init__()
        self.conv1 = dglnn.SAGEConv( 
            in_feats=in_feats, out_feats=hid_feats, feat_drop=0.2, aggregator_type='gcn')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, feat_drop=0.2, aggregator_type='mean')
        self.dropout =  nn.Dropout(dropout)
    
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = self.dropout(F.relu(h))
        h = self.conv2(graph, h)
        return h 
		
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
		
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def f1Score(tensor,label):
    T=F.softmax(tensor,dim=1)
    pre=[]
    for i in range(0,len(tensor)):
        if T[i][0]>=T[i][1]:
            pre.append(0)
        else:
            pre.append(1)
    nodeList=label.numpy().tolist()
    f1score=f1_score(pre,nodeList,average='micro')
    return f1score




model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels) 
opt = torch.optim.Adam(model.parameters())

best_val_acc = 0
Loss_train=[]
Loss_validation=[]
Loss_test=[]
Acc_validation=[]
Acc_test=[]
f1_test=[]
f1_validation=[]
for epoch in range(200): 
    print('Epoch {}'.format(epoch))
    model.train()
    # forward function
    logits = model(bg, node_features)
    # loss
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
    Loss_train.append(loss.item())
    
    loss1 = F.cross_entropy(logits[valid_mask], node_labels[valid_mask])
    Loss_validation.append(loss1.item())
    f1=f1Score(logits[valid_mask],node_labels[valid_mask])
    f1_validation.append(f1)
      
    loss2 = F.cross_entropy(logits[test_mask], node_labels[test_mask])
    Loss_test.append(loss2.item())
    f11=f1Score(logits[test_mask],node_labels[test_mask])
    f1_test.append(f11)
    
    # validation accuracy
    acc = evaluate(model, bg, node_features, node_labels, valid_mask)
    Acc_validation.append(acc)
    
    acc1 = evaluate(model, bg, node_features, node_labels, test_mask)
    Acc_test.append(acc)
    
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    
x=list(range(0,200))
plt.plot(x,Loss_validation,label='validation_loss')
plt.plot(x,Loss_test,label='test_loss')
plt.plot(x,Loss_train,label='train_loss')

plt.plot(x,f1_test,label='validation_f1')
plt.plot(x,f1_validation,label='test_f1')

plt.legend()
plt.show()

Acc_test[-1:]
Loss_validation[-1:]
Loss_test[-1:]
Acc_validation[-1:]
f1_test[-1:]
f1_validation[-1:]


##########################SMOTED Data############################
######## There is not much difference between original data and smoted data###########

!pip install imbalanced-learn
import imblearn
from imblearn.over_sampling import SMOTE

dfnode['word']=dfnode['tag'].apply(lambda x: x.replace('#',''))

temp=[]
for word in dfnode['word']:
    em=glove[word].tolist()
    temp.append(em)

dfnode['embeddings']=temp

# new dfkey
l=list(dflink.v1)+list(dflink.v2)
s=set(l)
len(s)
l=list(s)
dfkey=pd.DataFrame()
dfkey['tag']=l
dfkey['key']=[i for i in range(0,len(dfkey))]

# a is the new dataset for nodes
a=dfkey.merge(dfnode)
dflink=dflink.merge(a,left_on='v1',right_on='tag')
dflink=dflink.merge(a,left_on='v2',right_on='tag')

dflinks=dflink[['v1','embeddings_x','v2','embeddings_y',\
                'rumor_x','not_rumor_x']].copy()
dflinks.rename(columns={'embeddings_x':'em_v1','embeddings_y':'em_v2',\
                       'rumor_x':'rumor_link','not_rumor_x':'not_rumor_link'},\
              inplace=True)

temp=[]
for i in range(0,len(dflinks)):
    t1=dflinks['rumor_link'][i]
    t2=dflinks['not_rumor_link'][i]
    t3=t1/(t1+t2)
    temp.append(t3)
    
dflinks['rate']=temp
dflinks['label']=dflinks['rate'].apply(lambda x: 1 if x>dflinks.rate.mean() else 0)
A=list(dflinks.em_v1)
A=pd.DataFrame(A)
B=list(dflinks.em_v2)
B=pd.DataFrame(B)

Smote2B=dflinks[['v1','v2','label']]
Smote2B=pd.concat([Smote2B,A],axis=1)
Smote2B=pd.concat([Smote2B,B],axis=1)
name1=list(range(0,100))
name2=list(range(100,200))

x=Smote2B[name1+name2]
y=Smote2B['label']

smo=SMOTE(random_state=1,sampling_strategy=0.07)
x_smote,y_smote=smo.fit_sample(x,y)

len(a)
a.head(1)
temp=[]
for i in range(0,len(a)):
    t1=a.rumor[i]
    t2=a.not_rumor[i]
    t3=t1/(t1+t2)
    temp.append(t3)
a['rate']=temp
a['label']=a['rate'].apply(lambda x: 1 if x>a.rate.mean() else 0)

temp=[i for i in range(0,len(a))]
a['key']=temp

smotedf=pd.concat([x_smote,y_smote],axis=1)
em1=smotedf[name1]
em2=smotedf[name2]

# emlist1 and emlist2 is the embeddings after smote
temp=em1.values
emlist1=temp.tolist()
temp=em2.values
emlist2=temp.tolist()
emlist=list(a.embeddings)

NewWordEm=[]
ini=2877
src=[]
dst=[]
f1=0
f2=0
for i in range(0,len(emlist1)):
    if emlist1[i] not in NewWordEm and emlist2[i] not in NewWordEm:
        if emlist1[i] not in emlist and emlist2[i] not in emlist:
            NewWordEm.append(emlist1[i])
            NewWordEm.append(emlist2[i])
            t1=ini+NewWordEm.index(emlist1[i])
            t2=ini+NewWordEm.index(emlist2[i])
            src.append(t1)
            dst.append(t2)
         
        if emlist1[i] not in emlist and emlist2[i] in emlist:
            NewWordEm.append(emlist1[i])
            t1=ini+NewWordEm.index(emlist1[i])
            t2=emlist.index(emlist2[i])
            src.append(t1)
            dst.append(t2)
            f1=f1+1
            
        if emlist1[i] in emlist and emlist2[i] not in emlist:
            NewWordEm.append(emlist2[i])
            t1=emlist.index(emlist1[i])
            t2=ini+NewWordEm.index(emlist2[i])
            src.append(t1)
            dst.append(t2)
            f2=f2+1
        
        
        
        if i%10000==0:
            print(i)
print('\n')

df=Smote2B[['v1','v2']]
df=df.merge(a,left_on='v1',right_on='tag')
df=df[['v1','v2','key']]
df.rename(columns={'key':'key1'},inplace=True)
df=df.merge(a,left_on='v2',right_on='tag')
df=df[['v1','v2','key1','key']]
df.rename(columns={'key':'key2'},inplace=True)

newsrc=list(df.key1)+src
newdst=list(df.key2)+dst

wordEm=list(a.embeddings)+NewWordEm
src_tensor=torch.tensor(newsrc)
dst_tensor=torch.tensor(newdst)
g=dgl.graph((src_tensor,dst_tensor))
bg=dgl.to_bidirected(g)
bg=dgl.remove_self_loop(bg)
t=torch.tensor(wordEm)
bg.ndata['feat']=t
L=[1]*716
label=list(a.label)+L
bg.ndata['label']=torch.tensor(label)

import random
train=[]
validation=[]
test=[]
sample=[]
#Create percentage for train & validation set, you are able to change the percentage
def shuffle(train_per,validation_per):
    train_percentage = int(train_per * )
    validation_percentage = int(validation_per * len(dfkey))
    for i in range(0,len(dfkey)):
        if i<train_percentage:
            sample.append(1)
        if i>=train_percentage and i<validation_percentage:
            sample.append(2)
        if i>=validation_percentage:
            sample.append(3)
    random.shuffle(sample)
    for i in sample:
        if i==1:
            train.append(True)
            validation.append(False)
            test.append(False)
        if i==2:
            train.append(False)
            validation.append(True)
            test.append(False)
        if i==3:
            train.append(False)
            validation.append(False)
            test.append(True)

shuffle(0.3,0.7)

bg.ndata['train_mask']=torch.tensor(train)
bg.ndata['val_mask']=torch.tensor(validation)
bg.ndata['test_mask']=torch.tensor(test)
node_features=bg.ndata['feat']
node_labels=bg.ndata['label']
train_mask=bg.ndata['train_mask']
valid_mask=bg.ndata['val_mask']
test_mask=bg.ndata['test_mask']

n_features=node_features.shape[1]
n_labels=2

model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels) 
opt = torch.optim.Adam(model.parameters())

best_val_acc = 0
Loss_train=[]
Loss_validation=[]
Loss_test=[]
Acc_validation=[]
Acc_test=[]
f1_test=[]
f1_validation=[]
for epoch in range(200): 
    print('Epoch {}'.format(epoch))
    model.train()
    # forward function
    logits = model(bg, node_features)
    # loss
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
    Loss_train.append(loss.item())
    
    loss1 = F.cross_entropy(logits[valid_mask], node_labels[valid_mask])
    Loss_validation.append(loss1.item())
    f1=f1Score(logits[valid_mask],node_labels[valid_mask])
    f1_validation.append(f1)
      
    loss2 = F.cross_entropy(logits[test_mask], node_labels[test_mask])
    Loss_test.append(loss2.item())
    f11=f1Score(logits[test_mask],node_labels[test_mask])
    f1_test.append(f11)
    
    # validation accuracy
    acc = evaluate(model, bg, node_features, node_labels, valid_mask)
    Acc_validation.append(acc)
    
    acc1 = evaluate(model, bg, node_features, node_labels, test_mask)
    Acc_test.append(acc)
    
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    

x=list(range(0,200))
plt.plot(x,Loss_validation,label='validation_loss')
plt.plot(x,Loss_test,label='test_loss')
plt.plot(x,Loss_train,label='train_loss')

plt.plot(x,f1_test,label='validation_f1')
plt.plot(x,f1_validation,label='test_f1')

plt.legend()
plt.show()

Acc_test[-1:]
Loss_validation[-1:]
Loss_test[-1:]
Acc_validation[-1:]
f1_test[-1:]
f1_validation[-1:]
############################Dense FF model##########################
#######This model is used to compare with the GCN###################
from sklearn.model_selection import train_test_split
X = df.drop(columns=["label"])
Y = df["label"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

import keras
keras.__version__
from keras import models
from keras import layers
from keras.models import Sequential

train_X= df_train.drop(columns=["label"])
test_X= df_test.drop(columns=["label"])
train_label = df_train["label"]
test_label = df_test["label"]

x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.2)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(104,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(x_val, y_val))
					
					

f1Score=f1_score(Y_test, predictions, average='macro')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
#epochs = range(1, len(loss) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

###############################Graph######################
##############First Graph######################
dfNodes['max']=dfNodes[['neg','ne','po','m']].max(axis=1)
dfNodes['max_label']=np.argmax(np.array(dfNodes[['neg','ne','po','m']]),axis=1)

dfNodes["sent_rumor"] = dfNodes["rumor_group"]*10 + dfNodes["max_label"]
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([10, 11, 12, 13], 'Rumor')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([0], 'Negative')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([1], 'Netural')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([2], 'Positive')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([3], 'Mix')

dfNodes["sent_group"]=np.argmax(np.array(dfNodes[['neg','ne','po','m']]),axis=1)
dfNodes["rumor_group"]=np.argmax(np.array(dfNodes[['not_rumor','rumor']]),axis=1)

dfNodes["sent_rumor"]=np.argmax(np.array(dfNodes[['neg','ne','po','m']]),axis=1)
dfNodes["sent_rumor"] = dfNodes.apply(lambda x: x["rumor_group"] if x["rumor_group"]==1 else x["sent_rumor"], axis=1)

dfNodes["sent_rumor_prob"]=dfNodes[['neg','ne','po','m']].max(axis=1)/(dfNodes["neg"]+dfNodes["ne"]+dfNodes["po"]+dfNodes["m"])
dfNodes["sent_rumor_prob"] = dfNodes.apply(lambda x: x["rumorprob"] if x["rumor_group"]==1 else x["sent_rumor_prob"], axis=1)
dfNodes.to_csv('dfNodes.csv')

dfLinks["sent_rumor"] = dfLinks["rumor_group"]*10 + dfNodes["max_label"]
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([10, 11, 12, 13], 'Rumor')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([0], 'Negative')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([1], 'Netural')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([2], 'Positive')
dfLinks["sent_rumor"] = dfLinks["sent_rumor"].replace([3], 'Mix')

dfLinks["sent_group"]=np.argmax(np.array(dfLinks[['neg','ne','po','m']]),axis=1)
dfLinks["rumor_group"]=np.argmax(np.array(dfLinks[['not_rumor','rumor']]),axis=1)

dfLinks["sent_rumor"]=np.argmax(np.array(dfLinks[['neg','ne','po','m']]),axis=1)
dfLinks["sent_rumor"] = dfLinks.apply(lambda x: x["rumor_group"] if x["rumor_group"]==1 else x["sent_rumor"], axis=1)
dfLinks["sent_rumor_prob"]=dfLinks[['neg','ne','po','m']].max(axis=1)/(dfLinks["neg"]+dfLinks["ne"]+dfLinks["po"]+dfLinks["m"])
dfLinks["sent_rumor_prob"] = dfLinks.apply(lambda x: x["rumorprob"] if x["rumor_group"]==1 else x["sent_rumor_prob"], axis=1)

########Code below is using R kernel#########
install.packages('networkD3')
library(networkD3)

dfLinks$max <- paste(dfLinks$max, dfLinks$max_label, sep = add)
dfNodes <-  dfNodes[sample(nrow(dfNodes), 300), ]

color <- c("orange")
linkcolor <- color[(dfLinks$max_label) +1]

dfLinks$sent_rumor[dfLinks$sent_rumor == 10] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 11] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 12] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 13] <- "black"
dfLinks$sent_rumor[dfLinks$sent_rumor == 0] <- "lightblue" #negative
dfLinks$sent_rumor[dfLinks$sent_rumor == 1] <- "lightgreen" #neutral
dfLinks$sent_rumor[dfLinks$sent_rumor == 2] <- "red" #positive
dfLinks$sent_rumor[dfLinks$sent_rumor == 3] <- "orange"

dfNodes$sent_rumor[dfNodes$sent_rumor == 10] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 11] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 12] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 13] <- "black"
dfNodes$sent_rumor[dfNodes$sent_rumor == 0] <- "lightblue"
dfNodes$sent_rumor[dfNodes$sent_rumor == 1] <-  "lightgreen" 
dfNodes$sent_rumor[dfNodes$sent_rumor == 2] <- "red"
dfNodes$sent_rumor[dfNodes$sent_rumor == 3] <- "orange"
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([10, 11, 12, 13], 'Rumor')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([0], 'Negative')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([1], 'Netural')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([2], 'Positive')
dfNodes["sent_rumor"] = dfNodes["sent_rumor"].replace([3], 'Mix')

library(visNetwork)
library(igraph)
library(igraphdata)
library(stringr)
library(rpart)
library(sparkline)


Newnodes <- data.frame(id=dfNodes$tag, 
                       label = dfNodes$tag, 
                       #neutral: weight - 
                       value = dfNodes$sent_rumor_prob*10,
                       group = paste("Group",dfNodes$sent_rumor_prob),
                       
                       title = dfNodes$tag,
                       color = dfNodes$sent_rumor
                       )



Newedages <- data.frame(from = dfLinks$v1,
                        to = dfLinks$v2,
                        
                        value = dfLinks$sent_rumor_prob/100,
                        label = paste("weight",dfLinks$max,sep = "-"),
                        width = dfLinks$sent_rumor_prob/100, 
                        width = 0.1, 
                        color = dfLinks$sent_rumor
                        )


network <- visNetwork(Newnodes, Newedages, height = "1400px", width = "100%",
           main = "Sent_Rumor_Network Graph") %>%
  visOptions(highlightNearest = TRUE)%>%
  visInteraction(navigationButtons = TRUE)%>%
  visOptions(manipulation = TRUE)

visSave(network, file = "sent_rumor_network.html", background = "white")


######################Second Graph############################
# assign rumorprob_ proper weights for forcenetworkD3 graphing later.
dfNodes_highrumor$rumorprob_=10*round(round(100*dfNodes_highrumor$rumorprob)/20)
#summary(dfNodes_highrumor$rumorprob_)
dfLinks_highrumor$linkcolor=round(round(100*dfLinks_highrumor$rumorprob)/70)+1
#summary(dfLinks_highrumor$linkcolor)


library(networkD3)
# Plot
colors=JS('d3.scaleOrdinal().domain(["1", "2", "3"]).range(["#000000", "#111111"])')

p = forceNetwork(Links = dfLinks_highrumor, Nodes = dfNodes_highrumor,
            Source = "key1", Target = "key2",
            Value = "max", NodeID = "tag", opacity = 0.8, fontSize = 50,Group='rumor_group',Nodesize='rumor')

saveNetwork(p, file='BGraph.html', selfcontained = TRUE)

p=forceNetwork(Links = dfLinks_highrumor, Nodes = dfNodes_highrumor,
            Source = "key1", Target = "key2",
            Value = "max", NodeID = "tag", opacity = 0.9, fontSize = 50,Group='rumor_group',Nodesize='rumorprob_', colourScale = JS(ColourScale),linkColour=color)

saveNetwork(p, file='DGraph.html', selfcontained = TRUE)

library(dplyr)
library(htmltools)
color<-dfLinks_highrumor$color

# Df:Edges with rumor prob > 0.06. 409 nodes out of 4000+. 2074 edges
# Node Size: Rumor prob
# Node color: Purple: Negtive;  Orange: Positive
# Edge color: Rumor prob pink:rumor prob>0.5, blue:rumor prob<0.3
# Edge width: # of connections

ColourScale <- 'd3.scaleOrdinal()
            .domain(["lions", "tigers"])
           .range(["#694489","#FF6900",]);'

edgecolor<-colorRampPalette(c('#FFF00','FF0000'),bias=dfLinks_highrumor$linkcolor)
edgecolor<-sapply(dfLinks_highrumor$linkcolor)

browsable(
  tagList(
    tags$head(
      tags$style('
        body{background-color: #DAE3F9 !important}
        .nodetext{fill: #000000}
        .legend text{fill: #FF0000}
      ')
    ),
    forceNetwork(Links = dfLinks_highrumor, Nodes = dfNodes_highrumor,
            Source = "key1", Target = "key2",
            Value = "max", NodeID = "tag", opacity = 0.9, fontSize = 50,Group='rumor_group',Nodesize='rumorprob_', colourScale = JS(ColourScale),linkColour=color)
  )
)
