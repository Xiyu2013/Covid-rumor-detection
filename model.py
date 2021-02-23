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