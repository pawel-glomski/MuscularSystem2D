from Methods.Genetic import Genetic
from Methods.DQN import DQN
from Methods.DDPG import DDPG
from Environment import Environment
import numpy as np


def main():
    # np.random.seed(0)
    # method = Genetic(2, modelPath='drive/My Drive/models/GeneticBestModel')
    # method = DQN('drive/My Drive/models/DQNBestModel')
    method = DDPG(True)

    env = Environment(method.bodiesNum)
    maxCumReward = -float('inf')
    running = True
    while running:
        cumRewards = np.zeros(method.bodiesNum)
        done = False
        states = env.reset()
        while not done and running:
            rawActions = method.predict(states)
            newStates, rewards, done = env.step(method.toEnvActions(rawActions))
            cumRewards += rewards
            # if done and cumRewards.max() > maxCumReward:
            #     method.save(np.argmax(cumRewards), cumRewards.max(), env.episode)
            #     maxCumReward = cumRewards.max()
            # method.train(states, newStates, rawActions, rewards, cumRewards, done)
            states = newStates
            running = env.render(60)


if __name__ == "__main__":
    main()
