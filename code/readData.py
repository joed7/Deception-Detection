import os
import pickle

fake = '../data/hotelF-train.txt'
#fake='../data/a_neg'
f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), fake)))

real = '../data/hotelT-train.txt'
#real='../data/a_pos'
r = open(os.path.abspath(os.path.join(os.path.dirname(__file__), real)))

reviews={}
reviews['F']=[]
reviews['T']=[]

def readData():

    for i in f.readlines():
        i=i.strip()
        if len(i) == 0:
            continue
        #print i[0:6]
        text = i.lower()
        #print text
        reviews['F'].append(text)
    
    for i in r.readlines():
        i=i.strip()
        #print i[0:6]
        text = i.lower()
        #print text
        reviews['T'].append(text)
    
def createPickle(name,dict):
    fileObject = open(name,'wb') 
    pickle.dump(dict,fileObject)   
    fileObject.close()    

readData()

createPickle('../data/dataset', reviews)
#createPickle('../data/sentiment', reviews)

f.close()
r.close()
print('done')