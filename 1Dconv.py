import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape,UpSampling2D, Dropout
from keras import backend as K
import numpy as np
import h5py
from threadSafeIter import threadsafe_generator
from sklearn.preprocessing import MinMaxScaler

whatToEncode = "Is"
filename= "simSeriesData.h5"
chunckLength= 1
nEpochs= 5
trainPer = 0.9 #Approximate - depends on the divisibility of the total number of samples and

@threadsafe_generator
def generateTrainingDataFromFile(file="./simData.h5"):    
    
    with h5py.File(file,'r') as f:
        n = int((f["deltas"].shape[0]*trainPer)//chunckLength)
        for i in range(0,n):
            data = f['deltas'][i*chunckLength:(i+1)*chunckLength]
            if whatToEncode == 'Is':
                data = np.sin(data)
            data = data.reshape(data.shape[0],data.shape[1],1,1)
            yield (data,data)


@threadsafe_generator
def generateValidationDataFromFile(file="./simData.h5"):
    with h5py.File(file,'r') as f:
        ni = int((f["deltas"].shape[0]*trainPer)//chunckLength)
        nf = (f["deltas"].shape[0])//chunckLength
        for i in range(ni,nf):
            data = f['deltas'][i*chunckLength:(i+1)*chunckLength]
            if whatToEncode == 'Is':
                data = np.sin(data)
            data = data.reshape(data.shape[0],data.shape[1],1,1)
            yield (data,data)
        



with h5py.File(filename,'r') as f:
    nPoints,n = f['deltas'].shape

model = Sequential()

#Encode
model.add(Conv2D(5,5,input_shape=(n,1,1),padding="same"))
model.add(keras.layers.LeakyReLU())

model.add(Dropout(0.2))

model.add(Conv2D(5,5,padding="same"))
model.add(keras.layers.LeakyReLU())

model.add(MaxPooling2D((500,1)))

model.add(Conv2D(10,5,padding="same"))
model.add(keras.layers.LeakyReLU())

model.add(Dropout(0.2))

model.add(Conv2D(10,5,padding="same"))
model.add(keras.layers.LeakyReLU())

model.add(MaxPooling2D((500,1)))

model.add(Flatten())

model.add(Dense(40,input_shape=(40,)))

#Decode

model.add(Dense(40,input_shape=(40,)))

model.add(Reshape((4,1,10),input_shape=(40,)))


model.add(UpSampling2D((500,1)))

model.add(Conv2D(10,5,padding="same"))
model.add(keras.layers.LeakyReLU())

model.add(Dropout(0.2))

model.add(Conv2D(10,5,padding="same"))
model.add(keras.layers.LeakyReLU())

model.add(UpSampling2D((500,1)))

model.add(Conv2D(5,5,padding="same"))
model.add(keras.layers.LeakyReLU())

model.add(Dropout(0.2))

model.add(Conv2D(1,5,padding="same"))


model.compile(optimizer=keras.optimizers.Adam(),loss="mean_squared_error",
             metrics = ['accuracy'])


model.fit_generator(generateTrainingDataFromFile(filename),
                    verbose=1,
                    steps_per_epoch=(nPoints*trainPer)//(chunckLength*nEpochs),
                    epochs=nEpochs,
                    workers=3,
                    validation_data=generateValidationDataFromFile(filename),
                    validation_steps =(nPoints)//(chunckLength)-(nPoints*trainPer)//(chunckLength*nEpochs))

model.save(f"./en_{whatToEncode}.h5")   

"""
TO USE THE TRAINED MODEL
from keras.models import load_model
model = load_model('path/to/model')
y_test = model.predict(test)"""

