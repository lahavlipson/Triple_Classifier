import numpy as np
from sklearn.preprocessing import normalize
import random


vectorDict = {}
listOfRelations = []

#Initialize array of sets of triples. The set at each index contains all the triples 
#for the graph whose ID is the index
tripleArr = np.repeat(set(),36519)
for i in range(0,len(tripleArr)):
    tripleArr[i] = set()

setOfGoodTriples = set()
setOfBadTriples = set()

def setupVectors():
    with open("vectors.txt") as f:
        for line in f:
            line = line.rstrip()
            arr = line.split(' ')
           # print(arr[0])
            vectorDict[arr[0]] = list(map(float,arr[1:]))
  
def setUptripleArr():
    with open("deft-p2-amr-r2-training-ALL.triples") as f:
        for line in f:      
            trip = line.rstrip().split(' ')
            graphNum = int(trip[0])
            #print(graphNum)
            if trip[2][0:2] != 'op' and trip[2][0:4] != 'name':
                tripleArr[graphNum].add(tuple(trip[1:]))
                #print(int(trip[0]), tripleArr[int(trip[0])])

def setUpRelations():
    relationSet = set()
    for st in tripleArr:#For set in tripleArr
        for trip in st: #For triple in set
            relationSet.add(trip[1])
    global listOfRelations
    listOfRelations = list(relationSet)
    #print(listOfRelations)
#    
#    
def getOneHotEncodingOfRelation(rel):#Returns [0,...,0,1,0,...,0] 
    output = []
    #print(listOfRelations)
    for i in range(0,len(listOfRelations)):
        if listOfRelations[i] == rel:
            output.append(1);
        else:
            output.append(0)
    return np.array(output)

def setUpGoodTriples():
    for i in range(0,len(tripleArr)):
        setOfTriples = tripleArr[i]
        for trip in setOfTriples:
            setOfGoodTriples.add(trip)

def setUpBadTriples():
    for i in range(0,len(tripleArr)):
        listOfTriples = list(tripleArr[i])
        if (len(listOfTriples) >= 2):#Need two triples in order to create a good and bad one
            for j in range(0,len(listOfTriples)):
                firstTriple = listOfTriples[j]
                listExcludingFirstTriple = listOfTriples[0:j]+listOfTriples[(j+1):]
                secondTriple = listExcludingFirstTriple[random.randint(0,len(listExcludingFirstTriple)-1)]
                assert(len(secondTriple) == len(firstTriple)); assert(len(secondTriple) == 3)
                newBadTriple = firstTriple[0:2] + secondTriple[2:]
                setOfBadTriples.add(newBadTriple)
    
    
def turnTripleIntoGiantVector(triple):
    vec1 = vectorDict[triple[0]]
    vec2 = getOneHotEncodingOfRelation(triple[1])
    vec3 = vectorDict[triple[2]]
    return np.concatenate((vec1, vec2, vec3))
    
setupVectors()
setUptripleArr()
setUpRelations()
setUpGoodTriples()
setUpBadTriples()
print(len(setOfGoodTriples))
print(len(setOfBadTriples))#NOTE: There are many more bad tuples than good tuples because many of the good tuples are identical
print(turnTripleIntoGiantVector(('job', 'topic', 'that'))[45:55])#Testing

