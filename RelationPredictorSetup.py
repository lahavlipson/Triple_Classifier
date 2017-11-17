import numpy as np
import random
from TripleClassifier import setupVectors, isLineValid

    


def setUpTripleDict():#Creates an array of sets of triples
    tupleSet = set()
    relationSet = set()
    with open("deft-p2-amr-r2-training-ALL.triples") as f:
        
        for line in f:      
            trip = line.rstrip().split(' ')
            if isLineValid(line):
                
                lst = trip[1:]
                
                if lst[2] == "-":
                    lst[2] = "negative"
                            
                #Removes -03, -01, etc. for cases like Jump-02
                indexOfDash1 = lst[0].find('-')
                indexOfDash2 = lst[2].find('-')
                if indexOfDash1 >= 0:
                    lst[0] = lst[0][0:indexOfDash1]
                if indexOfDash2 >= 0:
                    lst[2] = lst[2][0:indexOfDash2] 
                    
                tupleSet.add(tuple(lst))
                relationSet.add(lst[1])
        return (tupleSet, relationSet)
    

def createRelationDict(triples, rels):
    relDict = {}    
   
    while len(triples) != 0:
        trip = triples.pop()
        rel = trip[1]
        if rel in rels:
            if rel in relDict:
                relDict[rel].append(trip)
            else:
                relDict[rel] = [trip]

    numberOfTripsPerRel = min(len(relDict[rels[0]]),len(relDict[rels[1]]),len(relDict[rels[2]]),len(relDict[rels[3]]))
    for rel in rels:
        random.shuffle(relDict[rel])
    for rel in rels:
        while len(relDict[rel]) > numberOfTripsPerRel:
            relDict[rel].pop()
        
    return relDict

    
def turnTripleIntoGiantVector(triple, vecDict):
    if triple[0] in vecDict and triple[2] in vecDict:
        vec1 = vecDict[triple[0]]
        vec2 = vecDict[triple[2]]
        return np.concatenate((vec1, vec2))
    else:
        return None            
    
   
    
        
def createTrainingData(dataDict, gloveVecs):
    for rel in dataDict:
        for trip in dataDict[rel]:
            vec = turnTripleIntoGiantVector(trip,gloveVecs)
            if vec is not None:
                assert(len(vec) == 100)
    




triples = list(setUpTripleDict()[0])
relationList = ["arg0","arg1","arg2","mod"]
print(len(triples))
rd = createRelationDict(triples, relationList)
print(rd['arg0'][1000])


vectorDict = setupVectors()
createTrainingData(rd, vectorDict)



    

