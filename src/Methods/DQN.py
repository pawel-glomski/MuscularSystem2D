import keras
import numpy as np
from typing import List
from collections import deque
import random


class DQN:
    def __init__(self, modelPath=None):
        self.bodiesNum = 1
        self.hipActionDisc = [-1, 1]
        self.kneeActionDisc = [-1, 1]
        self.ankleActionDisc = [-1, 1]
        self.actionSpace = []
        for hip1Action in self.hipActionDisc:
            for hip2Action in self.hipActionDisc:
                for knee1Action in self.kneeActionDisc:
                    for knee2Action in self.kneeActionDisc:
                        for ankle1Action in self.ankleActionDisc:
                            for ankle2Action in self.ankleActionDisc:
                                self.actionSpace.append([hip1Action, hip2Action, knee1Action, knee2Action, ankle1Action, ankle2Action])
        if modelPath is None:
            self.model = self._makeModel()
            self.targetModel = self._makeModel()
        else:
            self.model = keras.models.load_model(modelPath)
            self.targetModel = keras.models.load_model(modelPath)
        self.epsilon = 0.1
        self.epsilonMin = 0.01
        self.epsilonDecay = 1
        self.tau = 0.025
        self.replayBuffer = deque(maxlen=10000)

    def _makeModel(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(800, input_dim=30, activation='relu'))
        model.add(keras.layers.Dense(600, activation='relu'))
        model.add(keras.layers.Dense(len(self.actionSpace)))
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=0.005))
        return model

    def save(self, idx: int, cumReward: int, episode: int):
        print("!!!!!!!!!!!!!!!!!!!!! ", episode, "Saving, reward: %.2f" % cumReward, " !!!!!!!!!!!!!!!!!!!!!")
        self.model.save('drive/My Drive/models/DQNEp%d' % episode)

    def predict(self, statesArr):
        state = statesArr[0][0]
        self.epsilon = max(self.epsilonMin, self.epsilon*self.epsilonDecay)
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actionSpace))
        return np.argmax(self.model.predict(state)[0])

    def toEnvActions(self, rawAction):
        return [self.actionSpace[rawAction]]  # array, to match env's api

    def train(self, statesArr, newStates, rawActions, rewardsArr, cumRewards, done):
        self.remember(statesArr[0][0], rawActions, rewardsArr[0], newStates[0][0], done)
        self.replay()
        self.updateTarget()

    def remember(self, state, actionIdx, reward, newState, done):
        self.replayBuffer.append([state, actionIdx, reward, newState, done])

    def replay(self):
        batchSize = 16
        if len(self.replayBuffer) < batchSize:
            return

        samples = random.sample(self.replayBuffer, batchSize)
        for sample in samples:
            state, actionIdx, reward, newState, done = sample
            target = self.targetModel.predict(state)
            if done:
                target[0][actionIdx] = reward
            else:
                futureQ = max(self.targetModel.predict(newState)[0])
                target[0][actionIdx] = reward + (futureQ * 0.99)
            self.model.fit(state, target, epochs=1, verbose=0)

    def updateTarget(self):
        weights = self.model.get_weights()
        targetWeights = self.targetModel.get_weights()
        for i in range(len(targetWeights)):
            targetWeights[i] = weights[i] * self.tau + targetWeights[i] * (1 - self.tau)
        self.targetModel.set_weights(targetWeights)
