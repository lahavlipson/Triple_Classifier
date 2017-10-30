from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)


numOfData = 277500
dataDim = 145




dataset = np.zeros(numOfData*(dataDim+1))
dataset.shape = (numOfData, dataDim+1)
lineNum = 0
with open("/Users/lahavlipson/Research/Triple_Classifier/trainingData.txt") as f:
    for line in f:
        lArr = line.split(',')
        assert(len(lArr)==146)
        dataset[lineNum] = lArr    
        lineNum+=1

#Shuffles rows of data
np.random.shuffle(dataset)

# split into input (X) and output (Y) variables for training and testing
X_train = dataset[0:int(numOfData*0.8),0:dataDim]
Y_train = dataset[0:int(numOfData*0.8),dataDim]
X_test = dataset[int(numOfData*0.8):,0:dataDim]
Y_test = dataset[int(numOfData*0.8):,dataDim]


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
model.fit(X_train, Y_train, epochs=25, batch_size=10)

# calculate predictions
predictions = model.predict(X_test)

# round predictions
rounded = [round(x[0]) for x in predictions]

#Calculate accuracy on test case
numMatch = 0
for i in range(int(numOfData*0.2)):
    numMatch += int(rounded[i] == Y_test[i])
print("Accuracy on test:", numMatch/(numOfData*0.2))