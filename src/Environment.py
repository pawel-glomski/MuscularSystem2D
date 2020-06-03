from Box2D.Box2D import b2Body, b2EdgeShape, b2FixtureDef, b2Vec2, b2World, b2_pi
from typing import List
from Agent import Agent
import numpy as np
import pandas as pd
import Utils
import keras
import Display


class Environment:
    def __init__(self, agentsNum: int, recordResults: bool = True):
        self._makeWorld()
        self.agents = [Agent(self.world) for i in range(0, agentsNum)]
        self.episode = 0
        self.episodeTime = 16  # 50 bylo w ddpg
        self.time = 0
        self.timestep = 1.0/60
        self.record = recordResults
        self.resultRecord = {'Episode': [],
                             'MaxX': [],
                             'CumulativeReward': [],
                             'TimeAlive': []}
        self.randomImpulse = b2Vec2(0, 0)

    def _makeWorld(self):
        self.world = b2World(gravity=(0, -10), doSleep=True)
        self.MinHeight = 0.7
        self.ground_body = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
                                  categoryBits=0x0002, maskBits=0x0004),
            position=(0, 0),
            userData='ground'
        )
        self.helperEdge1 = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(0, -1e5), (0, 1e5)]),
                                  categoryBits=0x0004, maskBits=0x0002),
            position=(0, 0)
        )
        self.helperEdge2 = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, self.MinHeight), (1e5, self.MinHeight)]),
                                  categoryBits=0x0004, maskBits=0x0002),
            position=(0, 0)
        )

    def step(self, actionsArr: List[List[float]]) -> ((List[List[float]], List[bool]), List[float], bool):  # states, rewards, done?
        if self.time == 0:
            print("Starting episode %d" % self.episode + "... ", end="", flush=True)

        self._tick(actionsArr)
        stateArr, rewardArr = self._evalStates()
        done = self.time >= self.episodeTime or np.count_nonzero(stateArr[1]) == 0

        if done:
            agentsSorted = sorted(self.agents, key=lambda act: act.cumReward, reverse=True)
            print("Finished! ", "Best: %.2f, Worst: %.2f" % (agentsSorted[0].cumReward, agentsSorted[-1].cumReward),
                  "Time alive of best: %.2f, of worst: %.2f" % (agentsSorted[0].timeAlive, agentsSorted[-1].timeAlive))
            self._recordResults(agentsSorted)
        return (stateArr, rewardArr, done)

    def _tick(self, actionsArr: List[List[float]]):
        for agent, actions in zip(self.agents, actionsArr):
            if agent.active:
                agent.applyActions(actions)
        self.world.Step(self.timestep, 5, 10)
        self.time += self.timestep

        applyImpulse = np.random.random() <= self.timestep*1.5
        if applyImpulse:
            self.randomImpulse = b2Vec2(np.random.choice([-1, 1]) * np.random.normal(5, 1), 0)
        for agent, actions in zip(self.agents, actionsArr):
            if agent.active:
                agent.applyActions(actions)
                agent.bones['torso'].ApplyLinearImpulse(impulse=self.randomImpulse, point=(0.6, 0), wake=True)
        self.randomImpulse *= 0.5

    def _evalStates(self) -> ((List[List[float]], List[bool]), List[float]):  # (state, reward)
        rewardArr = [0] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if agent.active:
                agentPos = agent.getRootPos()
                if agentPos.y >= self.MinHeight:
                    agent.hp = min(agent.hp + 1.5*self.timestep, 1)
                else:
                    agent.hp = max(agent.hp - self.timestep, 0)
                    #     # or abs(agent.bones['torso'].angle + b2_pi/2) > b2_pi/3
                    #     # 9 - time of episode used to figure out average worst case reward
                    #     rewardArr[i] = (self.episodeTime / 9) * -200 * (1-agent.timeAlive / self.episodeTime)  # not good for Q-value related algorithms
                    #     # rewardArr[i] = -300
                    #     agent.deactivate()
                    # else:
                if agent.hp == 0:
                    rewardArr[i] = -100
                    agent.deactivate()
                else:
                    rewardArr[i] = self._calcReward(agent)
                agent.cumReward += rewardArr[i]
                agent.maxX = max(agent.maxX, agentPos.x)
                agent.timeAlive += self.timestep
        return (self._getStates(), rewardArr)

    def _calcReward(self, agent: Agent) -> float:
        torques = 0
        for joint in agent.joints:
            torques += joint.GetMotorTorque(60) ** 2  # hardcoded fps
        return (
            2*abs(agent.bones['torso'].linearVelocity.x) + agent.bones['torso'].linearVelocity.x +
            - torques*0.00001
            + 0.1
        )

    def _getStates(self):
        return ([a.getState() for a in self.agents], [a.active for a in self.agents])

    def reset(self) -> List[List[float]]:
        self.time = 0
        self.episode += 1
        off = b2Vec2(np.random.normal(0, 0.1), 0)  # same for each
        for a in self.agents:
            a.softReset(off)
        for _ in range(4):
            self.world.Step(self.timestep*2, 3, 6)
        return self._getStates()

    def _recordResults(self, agentsSorted):
        if self.record:
            self.resultRecord['Episode'].append(self.episode)
            self.resultRecord['MaxX'].append(agentsSorted[0].maxX)
            self.resultRecord['CumulativeReward'].append(agentsSorted[0].cumReward)
            self.resultRecord['TimeAlive'].append(agentsSorted[0].timeAlive)
            if self.episode % 10 == 9:
                pd.DataFrame(self.resultRecord, columns=['Episode', 'MaxX', 'CumulativeReward', 'TimeAlive']).to_csv('records.csv', index=False)

    def render(self) -> bool:
        return Display.clearDrawDisplay(self)
