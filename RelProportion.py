import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from TripleClassifier import setUpRelations, setUpTripleDict, setUpGoodTriples


def main(b): 
    #b = True to show proportion of relations in initial tripleData
    #b = False to show proportion of relations in wrong_triples.txt
    tripleDict = setUpTripleDict()
    listOfRelations = setUpRelations(tripleDict)  
    
    relDict = {}
    for r in listOfRelations:
        relDict[r] = 0
    
    
    if b:
        trips = setUpGoodTriples(tripleDict)
        
        for t in trips:
            for rel in relDict:
                if rel in "".join(t):
                    relDict[rel] += 1
    else:
        with open("wrong_triples.txt") as f:
            for line in f:
                for rel in relDict:
                    if rel in line:
                        relDict[rel] += 1
        
    
    objects = []
    y_pos = []
    performance = []
    numBars = 0
    keyList = list(relDict.keys())
    keyList.sort(reverse=True)
    for key in keyList:
        if (relDict[key] > 3000 and b) or (relDict[key] > 300 and not b):
            y_pos.append(numBars)
            performance.append(relDict[key])
            objects.append(key)
            numBars+=1
    
    print("relDict:",relDict)
    print(objects)
    plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, list(objects))
    plt.xlabel('Frequency')
    plt.title('Relation Frequency')    
    plt.show()
    
main(True)