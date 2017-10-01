import numpy as np
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

def setupVectors():#Creates dictionary of vectors and their embedding vectors
    with open("vectors.txt") as f:
        for line in f:
            line = line.rstrip()
            arr = line.split(' ')
           # print(arr[0])
            vectorDict[arr[0]] = list(map(float,arr[1:]))
  
def setUpTripleArr():#Creates an array of sets of triples
    with open("deft-p2-amr-r2-training-ALL.triples") as f:
        for line in f:      
            trip = line.rstrip().split(' ')
            graphNum = int(trip[0])
            if trip[2][0:2] != 'op' and trip[2][0:4] != 'name':
                lst = trip[1:]
                #Removes -03, -01, etc. for cases like Jump-02
                indexOfDash1 = lst[0].find('-')
                indexOfDash2 = lst[2].find('-')
                if indexOfDash1 >= 0:
                    lst[0] = lst[0][0:indexOfDash1]
                if indexOfDash2 >= 0:
                    lst[2] = lst[0][2:indexOfDash2]
                tripleArr[graphNum].add(tuple(lst))

def setUpRelations():#Creates a list of all the possible relations (About 100)
    relationSet = set()
    for st in tripleArr:#For set in tripleArr
        for trip in st: #For triple in set
            relationSet.add(trip[1])
    global listOfRelations
    listOfRelations = list(relationSet)


def getOneHotEncodingOfRelation(rel):#Returns [0,...,0,1,0,...,0] 
    output = []
    for i in range(0,len(listOfRelations)):
        if listOfRelations[i] == rel:
            output.append(1);
        else:
            output.append(0)
    return np.array(output)

def setUpGoodTriples():#Just compiles all of the triple from tripleArr into a set
    for i in range(0,len(tripleArr)):
        setOfTriples = tripleArr[i]
        for trip in setOfTriples:
            setOfGoodTriples.add(trip)

def setUpBadTriples():#Recombines triples to form bad ones
    for i in range(0,len(tripleArr)):
        listOfTriples = list(tripleArr[i])
        if (len(listOfTriples) >= 2):
        #Need two triples in order to create a good and bad one
            for j in range(0,len(listOfTriples)):
            #Creates triple where the first two components are unchanced 
            #and the third is from a different triple from the same graph
                firstTriple = listOfTriples[j]
                listExcludingFirstTriple = listOfTriples[0:j]+listOfTriples[(j+1):]
                secondTriple = listExcludingFirstTriple[random.randint(0,len(listExcludingFirstTriple)-1)]
                assert(len(secondTriple) == len(firstTriple)); assert(len(secondTriple) == 3)
                newBadTriple = firstTriple[0:2] + secondTriple[2:]
                setOfBadTriples.add(newBadTriple)
    
    
def turnTripleIntoGiantVector(triple):#Returns a vector that can be trained on. Returns None if the first or last word is not in the embedding
    if triple[0] in vectorDict and triple[2] in vectorDict:
        vec1 = vectorDict[triple[0]]
        vec2 = getOneHotEncodingOfRelation(triple[1])
        vec3 = vectorDict[triple[2]]
        return np.concatenate((vec1, vec2, vec3))
    else:
        return None
    
setupVectors()
setUpTripleArr()
setUpRelations()
setUpGoodTriples()
setUpBadTriples()
print(len(setOfGoodTriples))#Number tuples
print(len(setOfBadTriples))#NOTE: There are many more bad tuples than good tuples because many of the good tuples are identical
print(turnTripleIntoGiantVector(('job', 'topic', 'that'))[45:55])#Testing


count = 0
for trip in setOfGoodTriples:
    h = turnTripleIntoGiantVector(trip)
    if h is not None:
        count+=1
print(count)#This is the number of triples that contain a word not in the word embedding

