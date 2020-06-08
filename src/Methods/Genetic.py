from Environment import Environment
from collections import deque
from random import sample
from typing import List
from Body import Body
import tensorflow as tf
import numpy as np
import random
import keras
import os


class Genetic:
    def __init__(self, trainingBodiesNum, modelPath=None):
        self.bodiesNum = 1
        self.trainingModels = []
        if modelPath is not None:
            self.mainModel = keras.models.load_model(modelPath)
        else:
            self.mainModel = Genetic._makeModel()
        self.targetModel = Genetic._makeModel(self.mainModel, self.mainModel.get_weights())
        self.trainingEnv = Environment(trainingBodiesNum, epTime=2)
        self.pastBuffer = deque(maxlen=1000)
        self.targetActions = [np.zeros((1, 6))] * trainingBodiesNum
        self.currentActions = [np.zeros((1, 6))] * trainingBodiesNum
        self.mutation = 0
        self.learningStatesSize = 8
        self.initMutation = np.pi / 7
        self.mutationStep = (np.pi - 2*self.initMutation) / (self.learningStatesSize - 1)

    @staticmethod
    def _makeModel(base=None, wbs=None, lr=0.0005):
        if base is None:
            model = keras.Sequential()
            model.add(keras.layers.Dense(600, input_dim=30, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(400, activation='relu'))
            model.add(keras.layers.Dropout(0.1))
            model.add(keras.layers.Dense(6, activation='tanh'))
        else:
            model = keras.models.clone_model(base)
        if wbs is not None:
            model.set_weights(wbs)
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=lr))
        return model

    def save(self, idx: int, cumReward: int, episode: int):
        print("!!!!!!!!!!!!!!!!!!!!! ", episode, "Saving, reward: %.2f" % cumReward, " !!!!!!!!!!!!!!!!!!!!!")
        self.mainModel.save('drive/My Drive/models/GenEp%d' % episode)

    def predict(self, statesArr) -> List[List[float]]:
        return [self.mainModel.predict(statesArr[0])[0]]
        # return [[0] * 6 if active else None for model, state, active in zip(self.trainingModels, statesArr[0], statesArr[1])]

    def toEnvActions(self, rawAction):  # to match main's api
        return rawAction

    def train(self, statesArr, newStates, rawActions, rewardsArr, cumRewards, done):
        body: Body = statesArr[1][0]
        if (body.health > 0.85 and np.random.random() < 1/8) or len(self.pastBuffer) > 0:
            self.pastBuffer.append(body.getRealStates())
        if len(self.pastBuffer) >= 16:
            print('\nTraining...', end='\t', flush=True)
            for past in [self.pastBuffer[0], self.pastBuffer[7], self.pastBuffer[15]]:
                self.mutation = 0  # enable mutation
                learningStates = []
                cumRewards = np.zeros(len(self.trainingEnv.bodies))
                done = False
                states = self.trainingEnv.reset(initState=past)
                while not done:
                    self._mutate(states[0])
                    if len(learningStates) < self.learningStatesSize:
                        learningStates.append(states[0])
                    newStates, rewards, done = self.trainingEnv.step([actions[0] for actions in self.currentActions], mainEnv=False)
                    cumRewards += rewards
                    states = newStates
                    self.trainingEnv.render()
                self.learn(learningStates, cumRewards)
            self.updateTarget()
            print('Finished')
            self.pastBuffer.clear()

    def _mutate(self, states) -> bool:  # mutated?
        if self.mutation == 0:
            self.mutation = self.initMutation
            self.targetActions[0] = self.mainModel.predict(states[0]) if states[0] is not None else None
            i = 1
            for _ in range(2):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        for s3 in [-1, 1]:
                            for s4 in [-1, 1]:
                                for s5 in [-1, 1]:
                                    for s6 in [-1, 1]:
                                        self.targetActions[i] = np.reshape(np.random.uniform(0.95, 0.0, 6) * np.array([s1, s2, s3, s4, s5, s6]),
                                                                           (1, -1))
                                        i += 1
                                    # self.targetActions[i] = np.reshape(np.random.uniform(1, 0.6, 6) *
                                    #                                    np.random.choice([-1, 1], 6, True), (1, -1))
            # for i in range(len(self.targetActions)):
            #     self.targetActions[i] = np.reshape(np.random.uniform(-1, 1, 6), (1, -1))
        for i, (targetActions, state) in enumerate(zip(self.targetActions, states)):
            if state is not None:
                self.currentActions[i] = np.array(self.targetActions[0]) if i == 0 else (
                    targetActions * np.sin(self.mutation) + self.mainModel.predict(state) * (1 - np.sin(self.mutation)))
        if self.mutation != -1:
            self.mutation += self.mutationStep
            self.mutation = self.mutation if self.mutation <= np.pi - self.initMutation else -1

    def learn(self, learningStates, cumRewards):
        idxSorted = np.argsort(-cumRewards)  # - to reverse the order to descending
        if idxSorted[0] == 0:  # do not train if the best one is the current best one
            return False
        bestIdx = idxSorted[0]
        secondIdx = idxSorted[1]
        rRatio = (cumRewards[secondIdx] / cumRewards[bestIdx] / 2) if cumRewards[secondIdx] > 0 and cumRewards[bestIdx] != 0 else 0
        # print()
        targetActions = (self.targetActions[bestIdx] * (1 - rRatio) + self.targetActions[secondIdx] * rRatio) * 0.75
        # print(targetActions)
        targetActions += sum(np.array(self.targetActions)[idxSorted[:30]])/30 * 0.175
        # print(targetActions)
        targetActions -= sum(np.array(self.targetActions)[idxSorted[-30:]])/30 * 0.075
        # print(targetActions)
        targetActions = np.clip(targetActions, -1, 1)

        tau = self.initMutation
        for states in learningStates:
            state = states[bestIdx]
            currentActions = targetActions * np.sin(tau) + self.targetModel.predict(state) * (1-np.sin(tau))
            self.mainModel.fit(state, currentActions, epochs=1, verbose=0)
            tau += self.mutationStep
        return True

    def updateTarget(self):
        tau = 0.1
        weights = self.mainModel.get_weights()
        targetWeights = self.targetModel.get_weights()
        for i in range(len(targetWeights)):
            targetWeights[i] = targetWeights[i] * (1 - tau) + weights[i] * tau
        self.targetModel.set_weights(targetWeights)

    # unused for now, but may be in future
    @staticmethod
    def _selection(rewardsArr: List[float]):
        return [i for i, prob in enumerate(Genetic.popProbs(rewardsArr)) if np.random.rand() <= prob]

    @staticmethod
    def popProbs(rewardsArr: List[float]) -> np.ndarray:
        normalized = (np.array(rewardsArr) - np.max(rewardsArr))
        normalized = 4**((normalized - np.mean(normalized)) / np.std(normalized))
        normalized = normalized / np.sum(normalized)
        probs = np.zeros(len(rewardsArr))
        total = 0
        for idx in np.argsort(normalized):
            total += normalized[idx]
            probs[idx] = total
        return probs
