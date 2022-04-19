from joblib import load
import numpy as np
model = load("model.joblib")



def predict_fields(myArray):
    values = np.array(myArray)
    new = np.expand_dims(values,axis=0)
    myVal =  model.predict(new)
    print(myVal)
    return myVal
