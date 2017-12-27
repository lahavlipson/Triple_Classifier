from deps_tools import *
import enchant
from keras.models import model_from_json
from TripleClassifier import setupVectors
import numpy as np

def isNoun(pos):
	return pos[0] == 'N'

def isVerb(pos):
	return pos[0] == 'V'

def isAdj(pos):
	return pos[0] == 'J'

def isPunc(s):
    return s in list(".,/\"\'!")

def isEnglishWord(s):
    english = enchant.Dict("en_US")
    return english.check(s) and not isPunc(s)

def getSentenceAsStr(sent):
    output = ""
    for i in sent:
        output += sent[i].word + " "
    return output
        
def createVector(double, vecDict):
    if double[0] in vecDict and double[1] in vecDict:
        vec0 = vecDict[double[0]]
        vec1 = vecDict[double[1]]
        return np.concatenate((vec0, vec1))
    else:
        return None 
    
def predictRel(model, vectorDict,double):
    relChoices = ["arg0","arg1","arg2","mod","domain","location","mod","poss","time"]
    vec = createVector(double,vectorDict)
    if vec is None:
        return "NoChoice"
    predictions = model.predict(vec.reshape((1,100)))
    return relChoices[np.argmax(predictions)]


    
def readModel():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    return model



def buildAMRDict(model,vectorDict):
    wordDict = {}
    for i in sent:
        if isEnglishWord(sent[i].word):
            children = []
            for j in sent:
                if (sent[j].parent_id == sent[i].node_id):
                    children.append(sent[j].node_id)

            wordDict[sent[i].node_id] = (sent[i].word,children, sent[i].pos)            
    return wordDict


def getAMR(wordDict, index, depth, model, vectorDict):
    output = "("+wordDict[index][0]
    for child in wordDict[index][1]:
        if child != index and child in wordDict:
            double = (wordDict[index][0],wordDict[child][0])
            arg = "!ROC" #This current node is not an Open-Class word
            if isNoun(wordDict[index][2]) or isVerb(wordDict[index][2]) or isAdj(wordDict[index][2]):
                arg = predictRel(model, vectorDict, double)
            output += "\n" + "   "*depth + ":" + arg + getAMR(wordDict,child,depth+1, model, vectorDict)
    return output +")";



#Getting a sentence from the dataset
f = DepsFile(open("deft-p2-amr-r2-amrs-training-ALL-hyphen.deps",'r'))
sent = f.next_sentence()[0]
for i in range(249):#Sentence number 110. This is arbitrary (change it!)
	sent = f.next_sentence()[0]


#Get GloVe vectors
vectorDict = setupVectors()

#Get neural net
model = readModel()

#get dictionary containing each word and its child
wordDict = buildAMRDict(model,vectorDict)

#Find root of tree
rootID = -1
for i in range(1,len(sent)+1):
        if sent[i].node_id == sent[i].parent_id:
            rootID = sent[i].node_id


print("\n"+getAMR(wordDict, rootID, 1, model, vectorDict))
print("\n"+getSentenceAsStr(sent))


