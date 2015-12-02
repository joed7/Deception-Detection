import pickle
from random import randrange
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from _collections import defaultdict
from nltk.tag import pos_tag
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset={}
train={}
test={}
stopword={}
vec = DictVectorizer()
clf = None

def loadDataSet(name):
    global dataset,stopword
    
    fileObject = open(name,'r')  
    dataset = pickle.load(fileObject)  
    #print len(dataset['r'])
    
    sw=stopwords.words('english')
    stopword={}
    for i in sw:
        stopword[str(i).lower()]=1

def splitData():
    positive = dataset['T'][:]
    fake = dataset['F'][:]
    
    size = len(positive)
    tr_size = int(size*0.8)
    
    temp=[]
    for i in range(tr_size):
        random_index = randrange(0,len(positive))        
        temp.append(positive[random_index])
        positive.pop(random_index)
        
    train['T']=temp
    test['T']=positive
    
    #print(str(len(temp)) + ","+str(len(positive)) )
    
    temp=[]
    for i in range(tr_size):
        random_index = randrange(0,len(fake))        
        temp.append(fake[random_index])
        fake.pop(random_index)
        
    #print(str(len(temp)) +","+ str(len(fake)) )
        
    train['F']=temp
    test['F']=fake

def train_data():
    #train positive examples
    global vocab,totalF,totalR,dict,clf
    
    pos_docs=train['T']
    
    training_data=[]
    label_data=[]
    
    
    for pos_doc in pos_docs:
        #id-1174
        features=defaultdict(int)
        pos_doc=pos_doc[7:]
        sentences = sent_tokenize(pos_doc)
        
        #features['numsen']=len(sentences)
        total_words = 0
        
        for s in sentences:
            words = word_tokenize(s)
            tagged = pos_tag(words)
            total_words = total_words +len(words)
            
            for word,tag in tagged:
                if word.isalpha() and not isStopWord(word):
                    features[tag]=features[tag]+1
                    #features[word]=features[word]+1
        
        #features['avg_word'] = total_words * 1.0 /len(sentences)     
              
        training_data.append(features)
        label_data.append('T')
                     
    fake_docs=train['F']
    
    for fake_doc in fake_docs:

        features=defaultdict(int)
        fake_doc=fake_doc[7:]
        sentences = sent_tokenize(fake_doc)
       
        #features['numsen']=len(sentences)
        total_words = 0
              
        for s in sentences:
            words = word_tokenize(s)
            tagged = pos_tag(words)
            total_words = total_words +len(words)
                        
            for word,tag in tagged:
                if word.isalpha() and not isStopWord(word):
                    features[tag]=features[tag]+1
                    #features[word]=features[word]+1

        #features['avg_word'] = total_words * 1.0 /len(sentences)     
        training_data.append(features)   
        label_data.append('F')        
    
    feature_matrix = vec.fit_transform(training_data).toarray()     
    label_matrix = np.array(label_data)
    
    
    clf = SVC(kernel='linear',C=1)
    clf.fit(feature_matrix, label_matrix)
    
def test_data1():
    
    test_data=[]
    labeled_data=[]
    
    for (label,docs) in test.items():
        for doc in docs:
           
            tag_word=[]
            id = doc[0:7]
            doc = doc[7:]
            sentences = sent_tokenize(doc)
            
            features=defaultdict(int)
            #features['numsen']=len(sentences)
            total_words = 0            
            
            for s in sentences:
                words = word_tokenize(s)
                tagged = pos_tag(words)

                total_words = total_words +len(words)

                for word,tag in tagged:
                    if word.isalpha() and not isStopWord(word):
                        features[tag]=features[tag]+1
                        #features[word]=features[word]+1
            #features['avg_word'] = total_words * 1.0 /len(sentences)
            
            test_data.append(features)
            labeled_data.append(label)
            
            
    
    output = classify(test_data) 
    #print output
    return accuracy_score(output, np.array(labeled_data))
    #print('accuracy='+str(1.0*correct/tot))
    #return 1.0*correct/tot
    
def test_data2():
    
    fileObject = open('../data/test.txt', 'r')  
    inputData = fileObject.readlines()
    test_data=[]
   
   
    for doc in inputData:
        doc = doc.strip()
        
        tag_word=[]
        id = doc[0:7]
        doc = doc[7:]
        
        sentences = sent_tokenize(doc)
        features=defaultdict(int)       
   
        #features['numsen']=len(sentences)
        total_words = 0           
        
        for s in sentences:
            test_data = []
            words = word_tokenize(s)
            tagged = pos_tag(words)
            total_words = total_words +len(words)
            
            for word,tag in tagged:
                if word.isalpha() and not isStopWord(word):
                    features[tag]=features[tag]+1
                    #features[word]=features[word]+1        
        #features['avg_word'] = total_words * 1.0 /len(sentences)
        
        test_data.append(features)

        output = classify(test_data)
        #print output[0]
        print str(id)+'\t'+output[0]

            
def classify(test_data):
    
    test_matrix = vec.transform(test_data).toarray()     
    return clf.predict(test_matrix)


def isStopWord(word):
    return word in stopword     

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2) 
#loadDataSet('../data/sentiment')
result = 0
loadDataSet('../data/dataset')      

for i in range(10):
    splitData()  
    train_data()
    accuracy  = test_data1()
    result = result + accuracy
    print accuracy
    #test_data2()


print('Final Accuracy:'+str(result*1.0/10))    
#show_most_informative_features(vec,clf)
#test_data2()