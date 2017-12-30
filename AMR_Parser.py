from deps_tools import *
import enchant
from keras.models import model_from_json
from TripleClassifier import setupVectors
import numpy as np

def isOC(pos):
    return pos in ["JJ","JJR","JJS","MD","RB","RBR","RBS"] or pos[0] == "N" or pos[0] == "V"

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
    double = tuple([double[0].lower(),double[1].lower()])
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
        if isEnglishWord(sent[i].word) and sent[i].pos != "IN":
            children = []
            for j in sent:
                if (sent[j].parent_id == sent[i].node_id):
                    children.append(sent[j].node_id)
                
            if children != [] or isOC(sent[i].pos) or sent[i].word.isdigit():
                wordDict[sent[i].node_id] = (sent[i].word,children, sent[i].pos)            
    return wordDict


def getAMR(wordDict, index, depth, model, vectorDict,varNum, oneLine):
    output = "(x"+str(varNum.pop(0))+" / "+wordDict[index][0]
    for child in wordDict[index][1]:
        if child != index and child in wordDict:
            double = (wordDict[index][0],wordDict[child][0])
            arg = "!ROC" #This current node is not an Open-Class word
            if wordDict[child][2] == "CD":
                arg = "quant"
            elif wordDict[child][2][:3] == "NNP":
                arg = "name"
            elif isOC(wordDict[index][2]):
                arg = predictRel(model, vectorDict, double)  
            if not oneLine:    
                output += "\n" + "   "*(depth+1)
            output += " :" + arg + " " + getAMR(wordDict,child,depth+1, model, vectorDict, varNum, oneLine)        
    return output +")";




#Get GloVe vectors
vectorDict = setupVectors()

#Get neural net
model = readModel()



#Getting a sentence from the dataset
f = DepsFile(open("deft-p2-amr-r2-amrs-training-ALL-hyphen.deps",'r'))

sentenceNumber = 6

for j in range(sentenceNumber):
    sent = f.next_sentence()[0]
if j!=244:#sentence 244 has problems

    #If parent is proposition, set parent = grandparent
    for i in sent:
        if sent[i].parent_pos == "IN":
            sent[i].parent_id = sent[sent[i].parent_id].parent_id
            
    
    
    
    #get dictionary containing each word and its child
    wordDict = buildAMRDict(model,vectorDict)
    
    #Handle names
    for i in wordDict:
        if wordDict[i][2] == "NNP" or wordDict[i][2] == "NNPS":
            wordDict[i] = ("\""+wordDict[i][0]+"\"",wordDict[i][1],wordDict[i][2])
    
    #Find root of tree
    rootID = -1
    for i in range(1,len(sent)+1):
            if sent[i].node_id == sent[i].parent_id and sent[i].pos != "IN" and isEnglishWord(sent[i].word):
                rootID = sent[i].node_id
    
    if rootID > 0:
        varNum = list(range(100))
        amrStr = getAMR(wordDict, rootID, 0, model, vectorDict,varNum, False)
        print("\n"+amrStr)
        print("\n"+getSentenceAsStr(sent))
        
        
        

