import pickle
from random import randrange
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from _collections import defaultdict
from nltk.tag import pos_tag
import math


dataset={}
train={}
test={}
stopword={}

dict={}
dict['F']=defaultdict(int)
dict['T']=defaultdict(int)

vocab=0
totalF=0
totalR=0

def loadDataSet(name):
    global dataset
    fileObject = open(name,'r')  
    dataset = pickle.load(fileObject)  
    #print len(dataset['r'])
    
    sw=stopwords.words('english')
    stopword={}
    for i in sw:
        stopword[str(i).lower()]=1

def splitData():
    positive = dataset['T']
    fake = dataset['F']
    
    size = len(positive)
    tr_size = int(size*0.8)
    
    temp=[]
    for i in range(tr_size):
        random_index = randrange(0,len(positive))        
        temp.append(positive[random_index])
        positive.pop(random_index)
        
    train['T']=temp
    test['T']=positive
    
    print(str(len(temp)) + ","+str(len(positive)) )
    
    temp=[]
    
    for i in range(tr_size):
        random_index = randrange(0,len(fake))        
        temp.append(fake[random_index])
        fake.pop(random_index)
        
    print(str(len(temp)) +","+ str(len(fake)) )
        
    train['F']=temp
    test['F']=fake

def train_data():
    #train positive examples
    global vocab,totalF,totalR
    
    pos_docs=train['T']
    
    
    for pos_doc in pos_docs:
        #id-1174
        pos_doc=pos_doc[7:]
        sentences = sent_tokenize(pos_doc)
        for s in sentences:
            words = word_tokenize(s)
            for word in words:
                if word.isalpha() and not word in stopword:
                    dict['T'][word]= dict['T'][word]+1
                     
    fake_docs=train['F']
    
    for fake_doc in fake_docs:
        fake_doc=fake_doc[7:]
        sentences = sent_tokenize(fake_doc)
        for s in sentences:
            words = word_tokenize(s)
            for word in words:
                if word.isalpha() and not word in stopword:
                    dict['F'][word]= dict['F'][word]+1    
                    
    vocab = len(dict['T']) + len(dict['F'])                
    
    for (k,v) in dict['F'].items():
        totalF=totalF+v
            
    for (k,v) in dict['T'].items():
        totalR=totalR+v

def test_data1():
    
    correct =0
    tot=0
    for (label,docs) in test.items():
        for doc in docs:
            tag_word=[]
            id = doc[0:7]
            doc = doc[7:]
            sentences = sent_tokenize(doc)
           
            for s in sentences:
                words = word_tokenize(s)
                tagged = pos_tag(words)
                tag_word = tag_word +tagged
                #print tagged
            cat = classify(id, tag_word) 
            tot = tot +1
            if label == cat:
                correct = correct + 1
            
    print('accuracy='+str(1.0*correct/tot))
    
def test_data2():
    
    fileObject = open('../data/normal', 'r')  
    inputData = fileObject.readlines()
   
   
    for doc in inputData:
        doc = doc.strip()
        
        tag_word=[]
        id = doc[0:7]
        doc = doc[7:]
        
        sentences = sent_tokenize(doc)
       
        for s in sentences:
            words = word_tokenize(s)
            tagged = pos_tag(words)
            tag_word = tag_word +tagged
            #print tagged
        cat = classify(id, tag_word) 
        print id+'\t'+cat
           
def classify(id,words):
    
    #POS
    Rcontri=0
    Fcontri=0
    
    for (word,tag) in words:

        if word in stopword:
            continue
        
        if not word.isalpha():
             continue
       
        #if  (not tag.startswith('NN')) and (not tag.startswith('JJ')) and (not tag.startswith('VB')) and (not tag.startswith(' RB')):
            #continue

            
                
        rcount=dict['T'][word]
        fcount = dict['F'][word]
        
        #print(word+"->"+str(count))
        
        rcontri = (1+rcount)*1.0/(totalR+vocab)
        fcontri = (1+fcount)*1.0/(totalF+vocab)
       
        valr=math.log(rcontri)
        valf=math.log(fcontri)
        
        Rcontri = Rcontri+valr
        Fcontri = Fcontri + valf

    
    if math.fabs(Rcontri) < math.fabs(Fcontri):
        return 'T'
    else:
        return 'F'
      
loadDataSet('../data/dataset')      
#loadDataSet('../data/sentiment')
splitData()  
train_data()
test_data1()
#test_data2()