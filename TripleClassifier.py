import numpy as np
import random



def setupVectors():#Creates dictionary of vectors and their embedding vectors
    vectorDict = {}
    with open("vectors.txt") as f:
        for line in f:
            line = line.rstrip()
            arr = line.split(' ')
           # print(arr[0])
            vectorDict[arr[0]] = list(map(float,arr[1:]))
    return vectorDict

def isLineValid(line):
    arr = line.rstrip().split(' ')
    if arr[2][0:2] == 'op' or arr[2][0:4] == 'name':
        return False
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    for letter in alphabet:
        if ("-" + letter) in line:
            return False
    return True
  
def setUpTripleDict():#Creates an array of sets of triples
    tripDict = {}
    s = []#set()#Attempting to remove all duplicate triples. DOESN'T SEEM TO WORK
    with open("deft-p2-amr-r2-training-ALL.triples") as f:
        for line in f:
            s.append(line)
        
    for line in s:      
        trip = line.rstrip().split(' ')
        graphNum = int(trip[0])
        if graphNum not in tripDict:
            tripDict[graphNum] = set()
        if isLineValid(line):
            
            
            
            lst = trip[1:]
                        
            #Removes -03, -01, etc. for cases like Jump-02
            if lst[1] != 'polarity':
                indexOfDash1 = lst[0].find('-')
                indexOfDash2 = lst[2].find('-')
                if indexOfDash1 >= 0:
                    lst[0] = lst[0][0:indexOfDash1]
                if indexOfDash2 >= 0:
                    lst[2] = lst[2][0:indexOfDash2] 
                
            tripDict[graphNum].add(tuple(lst))
    return tripDict

def setUpRelations(tripDict):#Creates a list of all the possible relations (About 100)
    relationSet = set()
    for key in tripDict:#For set in tripleDict
        for trip in tripDict[key]: #For triple in set
            relationSet.add(trip[1])
    return list(relationSet)


def getOneHotEncodingOfRelation(rel,listOfAllRels):#Returns [0,...,0,1,0,...,0] 
    output = []
    for i in range(0,len(listOfAllRels)):
        if listOfAllRels[i] == rel:
            output.append(1);
        else:
            output.append(0)
    return np.array(output)

def setUpGoodTriples(tripDict):#Just compiles all of the triple from tripleDict into a set
    setOfGoodTriples = set()
    for i in tripDict:
        setOfTriples = tripDict[i]
        for trip in setOfTriples:
            setOfGoodTriples.add(trip)
    return setOfGoodTriples

def setUpBadTriples(tripDict):#Recombines triples to form bad ones
    setOfBadTriples = set()
    for i in tripDict:
        listOfTriples = list(tripDict[i])
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
    return setOfBadTriples
    
    
def turnTripleIntoGiantVector(triple, allRelations):#Returns a vector that can be trained on. Returns None if the first or last word is not in the embedding
    if triple[0] in vectorDict and triple[2] in vectorDict:
        vec1 = vectorDict[triple[0]]
        vec2 = getOneHotEncodingOfRelation(triple[1],allRelations)
        vec3 = vectorDict[triple[2]]
        return np.concatenate((vec1, vec2, vec3))
    else:
        return None
    
    
vectorDict = setupVectors()
tripleDict = setUpTripleDict()
print(tripleDict[103])
listOfRelations = setUpRelations(tripleDict)
setOfGoodTriples = setUpGoodTriples(tripleDict)
setOfBadTriples = setUpBadTriples(tripleDict)
print(len(setOfGoodTriples))#Number tuples
print(len(setOfBadTriples))#NOTE: There are many more bad tuples than good tuples because many of the good tuples are identical
print(turnTripleIntoGiantVector(('job', 'topic', 'that'),listOfRelations)[45:55])#Testing
count = 0
for trip in setOfGoodTriples:
    h = turnTripleIntoGiantVector(trip, listOfRelations)
    if h is not None:
        count+=1
print(count)#This is the number of triples that contain a word not in the word embedding

