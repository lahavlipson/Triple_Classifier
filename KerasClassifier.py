from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)


numOfData = 277500
dataDim = 145


# load pima indians dataset
dataset = np.loadtxt("/Users/lahavlipson/Research/Triple_Classifier/trainingData.txt", delimiter=",")

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
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=1, batch_size=1)

# calculate predictions
predictions = model.predict(X_test)

# round predictions
rounded = [round(x[0]) for x in predictions]

#Calculate accuracy on test case
numMatch = 0
for i in range(int(numOfData*0.2)):
    numMatch += int(rounded[i] == Y_test[i])
print("Accuracy on test:", numMatch/(numOfData*0.2))