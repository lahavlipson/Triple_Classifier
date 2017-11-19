from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import np_utils


dataDim = 100
numOfData = 53535
dataset = np.zeros(numOfData*(dataDim+2))
dataset.shape = (numOfData, dataDim+2)

lineNum=0

listOfTriples = []

with open("predictRelationTrain.txt") as f:
    for line in f:
        lst = line.rstrip().split(',')
        listOfTriples.append(lst[101])
        lst = lst[0:101]+[len(listOfTriples)-1]
        dataset[lineNum] = np.array(lst)
        lineNum+=1
assert(lineNum==numOfData)

#Shuffles rows of data
np.random.shuffle(dataset)

# split into input (X) and output (Y) variables for training and testing
X_train = dataset[:int(numOfData*0.8),:dataDim]
Y_train = dataset[:int(numOfData*0.8),dataDim]
X_test = dataset[int(numOfData*0.8):,:dataDim]
Y_test = dataset[int(numOfData*0.8):,dataDim:]

Y_train = np_utils.to_categorical(Y_train, 4)



model = Sequential()
model.add(Dense(100, input_dim=dataDim, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(4, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=10, batch_size=10)

# calculate predictions
predictions = model.predict(X_test)
    
relChoices = ["arg0","arg1","arg2","mod"]
numMatch = 0
with open("IncorrectRelPredictions.txt", 'w') as file_handler:
    for i in range(int(numOfData*0.2)):
        predCat = np.argmax(predictions[i])
        realCat = Y_test[i][0]
        if predCat == realCat:
            numMatch+=1
        else:
            trip = listOfTriples[int(Y_test[i][1])]
            file_handler.write(trip+" | Prediction: "+relChoices[predCat]+"\n")        
        
    acc = float("{:.2f}".format(numMatch/(numOfData*0.2)))
    print("Accuracy on test:", acc)