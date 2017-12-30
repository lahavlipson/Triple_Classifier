import numpy as np
import random
from TripleClassifier import setupVectors, isLineValid

    
relationList = ["arg0","arg1","arg2","mod","domain","location","mod","poss","time"]

def setUpTriples():
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
    

def createRelationDict(triples):
    relDict = {}    
   
    while len(triples) != 0:
        trip = triples.pop()
        rel = trip[1]
        if rel in relationList:
            if rel in relDict:
                relDict[rel].append(trip)
            else:
                relDict[rel] = [trip]

    numberOfTripsPerRel = -1
    for rel in relationList:
        numberOfTripsPerRel = max(numberOfTripsPerRel,len(relDict[rel]))
    #print("numberOfTripsPerRel:",numberOfTripsPerRel)
    for rel in relationList:
        while len(relDict[rel]) < numberOfTripsPerRel:
            relDict[rel] += relDict[rel]
        
        random.shuffle(relDict[rel])
        relDict[rel] = relDict[rel][:numberOfTripsPerRel]
            
        
    return relDict

    
def turnTripleIntoGiantVector(triple, vecDict):
    if triple[0] in vecDict and triple[2] in vecDict:
        vec1 = vecDict[triple[0]]
        vec2 = vecDict[triple[2]]
        return np.concatenate((vec1, vec2))
    else:
        return None            
    
   
    
        
def createTrainingData(dataDict, gloveVecs):
    output = []
    for rel in relationList:
        for trip in dataDict[rel]:
            
            
            xVec = turnTripleIntoGiantVector(trip,gloveVecs)
            if xVec is not None:
                assert(len(xVec) == 100)#Santiy check
                y = relationList.index(rel)
                #print(trip, y)
                vec = list(xVec)+[y,"-".join(trip)]
                assert(len(vec) == 102)
                output.append(vec)
    return output



def main():
    triples = list(setUpTriples()[0])
    rd = createRelationDict(triples)
    
    
#    for rel in relationList:
#        print(len(rd[rel]))

    vectorDict = setupVectors()
    trainingData = createTrainingData(rd, vectorDict)
    
    print(len(trainingData))
    
    with open("predictRelationTrain.txt", 'w') as file_handler:
        for datum in trainingData:
            strDatum = (','.join(str(v) for v in datum))+"\n"
            file_handler.write(strDatum)
             
main()


    