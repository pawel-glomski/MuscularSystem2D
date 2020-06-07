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
        self.targetActions = [[np.zeros((1, 6))]] * trainingBodiesNum
        self.currentActions = [np.zeros((1, 6))] * trainingBodiesNum
        self.mutation = 0
        self.learningStatesSize = 8
        self.initMutation = 1.0
        self.mutationDecay = 0.98

    @staticmethod
    def _makeModel(base=None, wbs=None, lr=0.005):
        if base is None:
            model = keras.Sequential()
            model.add(keras.layers.Dense(600, input_dim=30, activation='relu'))
            model.add(keras.layers.Dense(400, activation='relu'))
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

    def predict(self, statesArr: (List[List[float]], List[bool])) -> List[List[float]]:
        return [self.mainModel.predict(statesArr[0])[0]]
        # return [[0] * 6 if active else None for model, state, active in zip(self.trainingModels, statesArr[0], statesArr[1])]

    def toEnvActions(self, rawAction):  # to match main's api
        return rawAction

    def train(self, statesArr: (List[List[float]], List[Body]), newStates: (List[List[float]], List[bool]),
              rawActions: List[List[float]], rewardsArr: List[float], cumRewards: List[float], done: bool):
        body: Body = statesArr[1][0]
        if body.health > 0.95 or np.random.random() < body.health * 0.1:  # health <0,1>, do not save dying states
            self.pastBuffer.append(body.getRealStates())
        if done and len(self.pastBuffer) > 128:
            print('Training...', end='\t', flush=True)

            past = random.sample(self.pastBuffer, 1)[0]
            while True:
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
                if self.learn(learningStates, cumRewards):
                    break
            print('Finished')

    def _mutate(self, states) -> bool:  # mutated?
        if self.mutation == 0:
            self.mutation = self.initMutation
            for i in range(1, len(self.targetActions)):
                self.targetActions[i] = np.reshape(np.random.choice([-1, 1], 6, replace=True), (1, -1))
        for i, (targetActions, state) in enumerate(zip(self.targetActions, states)):
            if state is not None:
                mainActions = self.mainModel.predict(state)
                self.currentActions[i] = (targetActions * self.mutation + mainActions * (1 - self.mutation)) if i != 0 else mainActions
        self.mutation *= self.mutationDecay

    def learn(self, learningStates, cumRewards):
        idxSorted = np.argsort(-cumRewards)  # - to reverse the order to descending
        if idxSorted[0] == 0:  # do not train if the best one is the current best one
            return False
        bestIdx = idxSorted[0]
        secondIdx = idxSorted[1]
        rRatio = (cumRewards[secondIdx] / cumRewards[bestIdx] / 2) if cumRewards[secondIdx] > 0 else 0
        targetActions = self.targetActions[bestIdx] * (1 - rRatio) + self.targetActions[secondIdx] * rRatio

        tau = self.initMutation
        for states in learningStates:
            state = states[bestIdx]
            currentActions = targetActions * tau + self.targetModel.predict(state) * (1-tau)
            self.mainModel.fit(state, currentActions, epochs=1, verbose=0)
            tau *= self.mutationDecay
        self.updateTarget()
        return True

    def updateTarget(self):
        tau = 0.01
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
