import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten


def makeModel():
    model = Sequential()
    model.add(Dense(128, input_dim=28, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))

    return model
