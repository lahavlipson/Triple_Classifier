from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)


numOfData = 277500
dataDim = 145

listOfTriples = []


dataset = np.zeros(numOfData*(dataDim+2))
dataset.shape = (numOfData, dataDim+2)
lineNum = 0
with open("trainingData.txt") as f:
    for line in f:
        lArr = line.split(',')
        assert(len(lArr)==147)
        dataset[lineNum] = lArr[1:] + [lineNum]
        listOfTriples.append(lArr[0])
        lineNum+=1

#Shuffles rows of data
np.random.shuffle(dataset)

# split into input (X) and output (Y) variables for training and testing
X_train = dataset[0:int(numOfData*0.8),0:dataDim]
Y_train = dataset[0:int(numOfData*0.8),dataDim]
X_test = dataset[int(numOfData*0.8):,0:dataDim]
Y_test = dataset[int(numOfData*0.8):,dataDim:]


# create model
model = Sequential()
model.add(Dense(150, input_dim=dataDim, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=1, batch_size=10)

# calculate predictions
predictions = model.predict(X_test)

# round predictions
rounded = [round(x[0]) for x in predictions]

#Calculate accuracy on test case
numMatch = 0
with open("wrong_triples.txt", 'w') as file_handler:
    for i in range(int(numOfData*0.2)):
        if rounded[i] == Y_test[i][0]:
            numMatch+=1
        else:
            file_handler.write(listOfTriples[int(Y_test[i][1])]+"\n")        
    print("Accuracy on test:", numMatch/(numOfData*0.2))