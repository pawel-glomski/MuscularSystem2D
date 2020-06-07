from Methods.Genetic import Genetic
from Methods.DQN import DQN
from Methods.DDPG import DDPG
from Environment import Environment
import numpy as np


def main():
    # np.random.seed(0)
    # method = Genetic(100,
    #                  modelPath='drive/My Drive/models/Ep32'
    #                  )
    # method = DQN(
    #     # 'drive/My Drive/models/DQNEp1'
    # )
    env = Environment(1)
    method = []
    for i in range(4):
        if i == 0:
            method.append(DDPG())
            print('___Default DDPG___')
        elif i == 1:
            method.append(DDPG(useGaussianNoise=True))
            print('___Gaussian noise DDPG___')
        elif i == 2:
            method.append(DDPG(lateTraining=True))
            print('___Late training DDPG___')
        else:
            method.append(DDPG(useGaussianNoise=True, lateTraining=True))
            print('___Late training Gaussian noise DDPG___')
        running = True
        epCtr = 0
        maxCumReward = -float('inf')
        while running:
            cumRewards = np.zeros(method[i].agentsNum)
            done = False
            states = env.reset()
            while not done and running:
                rawActions = method[i].predict(states)
                newStates, rewards, done = env.step(method[i].toEnvActions(rawActions))
                cumRewards += rewards
                if done and cumRewards.max() > maxCumReward:
                    method[i].save(np.argmax(cumRewards), cumRewards.max(), env.episode)
                    maxCumReward = cumRewards.max()
                method[i].train(states, newStates, rawActions, rewards, cumRewards, done)
                states = newStates
                running = env.render()
            epCtr += 1
            if epCtr == 3000:
                running = False


if __name__ == "__main__":
    main()
