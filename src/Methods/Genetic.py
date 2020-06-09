from keras.initializers import RandomUniform
from keras.models import Sequential
from typing import List
import numpy as np
import tensorflow as tf
import keras
import os


class Genetic:
    def __init__(self, bodiesNum, modelPath=None):
        self.bodiesNum = bodiesNum
        self.usedModels = []
        self.swapModels = []
        if modelPath is not None:
            self.mainModel = keras.models.load_model(modelPath, compile=False)
            wbs = self.mainModel.get_weights()
            for _ in range(bodiesNum):
                self.usedModels.append(Genetic._makeModel(self.mainModel, wbs))
                self.swapModels.append(Genetic._makeModel(self.mainModel, wbs))
                self.usedModels[-1].set_weights(wbs)
                self.mutate(self.usedModels[-1], 0.05, 1)
        else:
            self.mainModel = Genetic._makeModel()
            for _ in range(bodiesNum):
                self.usedModels.append(Genetic._makeModel(self.mainModel))
                self.swapModels.append(Genetic._makeModel(self.mainModel))
                self.mutate(self.usedModels[-1], 0.05, 1)

    @staticmethod
    def _makeModel(base=None, wbs=None, lr=0.0005):
        if base is None:
            model = keras.Sequential()
            model.add(keras.layers.Dense(800, input_dim=30, activation='relu'))
            model.add(keras.layers.Dense(600, activation='relu'))
            model.add(keras.layers.Dense(6, activation='tanh'))
        else:
            model = keras.models.clone_model(base)
        if wbs is not None:
            model.set_weights(wbs)
        return model

    def save(self, idx: int, cumReward: int, episode: int):
        print("!!!!!!!!!!!!!!!!!!!!! ", episode, "Saving, reward: %.2f" % cumReward, " !!!!!!!!!!!!!!!!!!!!!")
        self.usedModels[idx].save('drive/My Drive/models/GenEp%d' % episode)

    def predict(self, statesArr):
        return [model.predict(state)[0] if body.active else None for model, state, body in zip(self.usedModels, statesArr[0], statesArr[1])]

    def toEnvActions(self, rawAction):  # to match main's api
        return rawAction

    def train(self, statesArr, newStates, rawActions, rewardsArr, cumRewards, done):
        if done:
            sel = self._selection(cumRewards)
            print("Number of selected parents = %d" % len(sel))
            self._swapModelsLists()
            for i, model in enumerate(self.usedModels):
                Genetic.crossover(model, np.random.choice(sel), np.random.choice(sel))
                Genetic.mutate(model, np.random.uniform(0.01, 0.1), 1)

    def _swapModelsLists(self):
        self.usedModels, self.swapModels = self.swapModels, self.usedModels

    def _selection(self, rewardsArr: List[float]):
        return [model for model, prob in zip(self.usedModels, Genetic.popProbs(rewardsArr)) if np.random.rand() <= prob]

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

    @staticmethod
    def crossover(model, base, other):
        if base is other:
            if base is not model:
                model.set_weights(base.get_weights())  # only mutate
        else:
            for i in range(len(base.layers)):
                wb1 = base.layers[i].get_weights()  # weights and biases
                wb2 = other.layers[i].get_weights()
                wb1[0] = wb1[0].T
                wb2[0] = wb2[0].T
                toSet = np.random.choice(np.arange(len(wb1[1])), int(len(wb1[1]) / 2))
                wb1[1][toSet] = wb2[1][toSet]
                wb1[0][toSet, :] = wb2[0][toSet, :]
                wb1[0] = wb1[0].T
                # for j in range(len(wb1)):
                #     wb1[j] += wb2[j]
                #     wb1[j] /= 2
                model.layers[i].set_weights(wb1)

    @staticmethod
    def mutate(model, mutationRate: float, mutationScale: float):
        rng = np.random.default_rng()
        for j, layer in enumerate(model.layers):
            new_weights_for_layer = []
            for weight_array in layer.get_weights():
                save_shape = weight_array.shape
                one_dim_weight = weight_array.reshape(-1)
                toChange = int(len(one_dim_weight) * mutationRate)
                change = np.concatenate((np.random.uniform(-0.175 * mutationScale, 0.175 * mutationScale, toChange),
                                         np.zeros(len(one_dim_weight) - toChange)))
                rng.shuffle(change)
                one_dim_weight += change
                new_weights_for_layer.append(one_dim_weight.reshape(save_shape))
            model.layers[j].set_weights(new_weights_for_layer)
