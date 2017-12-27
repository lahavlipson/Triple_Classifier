from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import model_from_json
from keras.utils import np_utils
from RelationPredictor import X_test, Y_test, numOfData



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")



# calculate predictions
predictions = model.predict(X_test)
    
relChoices = ["arg0","arg1","arg2","mod","domain","location","mod","poss","time"]
numMatch = 0
with open("CorrectRelPredictions.txt", 'w') as file_handler:
    for i in range(int(numOfData*0.2)):
        predCat = np.argmax(predictions[i])
        realCat = Y_test[i][0]
        if predCat == realCat:
            numMatch+=1              
        
    acc = float("{:.2f}".format(numMatch/(numOfData*0.2)))
    print("Accuracy on test:", acc)
    