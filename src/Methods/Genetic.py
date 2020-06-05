from Environment import Environment
from collections import deque
from random import sample
from typing import List
from Agent import Agent
import tensorflow as tf
import numpy as np
import random
import keras
import os


class Genetic:
    def __init__(self, trainingAgentsNum, modelPath=None):
        self.agentsNum = 1
        self.trainingModels = []
        if modelPath is not None:
            self.mainModel = keras.models.load_model(modelPath)
            wbs = self.mainModel.get_weights()
            for _ in range(trainingAgentsNum):
                self.trainingModels.append(Genetic._makeModel(0.001, self.mainModel, wbs))
                self.mutate(self.trainingModels[-1], 0.05, 1)
        else:
            self.mainModel = Genetic._makeModel(0.0001)
            for _ in range(trainingAgentsNum):
                self.trainingModels.append(Genetic._makeModel(0.01, self.mainModel))

        self.trainingEnv = Environment(trainingAgentsNum)
        self.pastBuffer = deque(maxlen=10000)

    def reset(self, env: Environment):
        env.reset()

    @staticmethod
    def _makeModel(lr, base=None, wbs=None):
        if base is None:
            model = keras.Sequential()
            model.add(keras.layers.Dense(800, input_dim=30, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01)))
            model.add(keras.layers.Dense(600, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01)))
            model.add(keras.layers.Dense(6, activation='tanh', kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01)))
        else:
            model = keras.models.clone_model(base)
        if wbs is not None:
            model.set_weights(wbs)
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=lr))
        return model

    def save(self, idx: int, cumReward: int, episode: int):
        print("!!!!!!!!!!!!!!!!!!!!! ", episode, "Saving, reward: %.2f" % cumReward, " !!!!!!!!!!!!!!!!!!!!!")
        self.trainingModels[idx].save('drive/My Drive/models/GenEp%d' % episode)

    def predict(self, statesArr: (List[List[float]], List[bool])) -> List[List[float]]:
        return [self.mainModel.predict(statesArr[0])[0]]
        # return [[0] * 6 if active else None for model, state, active in zip(self.trainingModels, statesArr[0], statesArr[1])]

    def toEnvActions(self, rawAction):  # to match main's api
        return rawAction

    def train(self, statesArr: (List[List[float]], List[bool], List[Agent]), newStates: (List[List[float]], List[bool]),
              rawActions: List[List[float]], rewardsArr: List[float], cumRewards: List[float], done: bool):
        agent: Agent = statesArr[2][0]
        if agent.hp > 0.9 or np.random.random() < agent.hp * 0.5:  # hp <0,1>, do not save dying states
            self.pastBuffer.append((statesArr[2][0].getRealStates(), agent.timeAlive))
        if done and len(self.pastBuffer) > 128:
            print('Training...', end='\t')

            past = random.sample(self.pastBuffer, 1)[0]
            while True:
                states = self.trainingEnv.reset(initState=past[0], initTime=past[1])
                trainingData = self._initTrainingModels(states[0][0])  # every agent has the same state, just pick the first one
                cumRewards = np.zeros(len(self.trainingEnv.agents))
                done = False
                while not done:
                    rawActions = [model.predict(state)[0] if active else None for model, state,
                                  active in zip(self.trainingModels, states[0], states[1])]
                    newStates, rewards, done = self.trainingEnv.step(rawActions, mainEnv=False)
                    cumRewards += rewards
                    states = newStates
                    self.trainingEnv.render()
                if self._trainMainModel(trainingData, cumRewards):  # break only if got better results
                    break
            print('Finished')

    def _initTrainingModels(self, initState):
        mainWBs = self.mainModel.get_weights()
        mainOutputs = self.mainModel.predict(initState)
        self.trainingModels[0].set_weights(mainWBs)  # current best to keep it
        self.trainingModels[0].targetOutput = mainOutputs
        for trainingModel in self.trainingModels[1:]:
            trainingModel.set_weights(mainWBs)
            trainingModel.targetOutput = np.reshape(np.random.choice([-1, 1], len(mainOutputs[0]), replace=True), (1, -1))
            trainingModel.fit(initState, trainingModel.targetOutput, epochs=3, verbose=0)
        return initState

    def _trainMainModel(self, initState, cumRewards):
        sortedIdxs = np.argsort(-cumRewards)  # - to reverse the order to descending
        if sortedIdxs[0] == 0:  # do not train if the best one is the current best one
            return False
        tau = cumRewards[sortedIdxs[1]] / cumRewards[sortedIdxs[0]] * 0.5
        targetOutputs = np.clip(self.trainingModels[sortedIdxs[0]].targetOutput * (1.0-tau) +
                                self.trainingModels[sortedIdxs[1]].targetOutput * tau,
                                -1, 1)
        self.mainModel.fit(initState, targetOutputs, epochs=1, verbose=0)
        return True

    def _parentModelsLists(self):
        self.trainingModels, self.parentModels = self.parentModels, self.trainingModels

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
