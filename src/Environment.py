from Box2D.Box2D import b2Body, b2EdgeShape, b2FixtureDef, b2Vec2, b2World, b2_pi
from typing import List
from Body import Body
import numpy as np
import pandas as pd
import Utils
import keras
import Display


class Environment:
    def __init__(self, bodiesNum: int, recordResults: bool = True, epTime: float = 16):
        self.episode = 0
        self.episodeTime = epTime
        self.time = 0
        self.timestep = 1.0/60
        self.randomImpulse = b2Vec2(0, 0)
        self.record = recordResults
        self.resultRecord = {'Episode': [],
                             'MaxX': [],
                             'CumulativeReward': [],
                             'TimeAlive': []}
        self.MinHeight = 0.75
        self._makeWorld()
        self.bodies = [Body(self.world) for i in range(0, bodiesNum)]

    def _makeWorld(self):
        self.world = b2World(gravity=(0, -10), doSleep=True)
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

    def step(self, actionsArr: List[List[float]], mainEnv: bool = True) -> ((List[List[float]], List[Body]), List[float], bool):  # states, rewards, done?
        if self.time == 0:
            if mainEnv:
                print("Starting episode %d" % self.episode + "... ", end="", flush=True)

        self._tick(actionsArr)
        stateArr, rewardArr = self._evalStates()
        done = self.time >= self.episodeTime or np.count_nonzero([a.active for a in self.bodies]) == 0

        if done and mainEnv:  # mainEnv = false when training in Genetic
            bodiesSorted = sorted(self.bodies, key=lambda act: act.cumReward, reverse=True)
            print("Finished! ", "Best: %.2f, Worst: %.2f" % (bodiesSorted[0].cumReward, bodiesSorted[-1].cumReward),
                  "Time alive of best: %.2f, of worst: %.2f" % (bodiesSorted[0].timeAlive, bodiesSorted[-1].timeAlive))
            self._recordResults(bodiesSorted)
        return (stateArr, rewardArr, done)

    def _tick(self, actionsArr: List[List[float]]):
        for body, actions in zip(self.bodies, actionsArr):
            if body.active:
                body.applyActions(actions)
        self.world.Step(self.timestep, 5, 10)
        self.time += self.timestep

        applyImpulse = np.random.random() <= self.timestep*1.5
        if applyImpulse:
            self.randomImpulse = b2Vec2(np.random.choice([-1, 1]) * np.random.normal(5, 1), 0)
        for body, actions in zip(self.bodies, actionsArr):
            if body.active:
                body.applyActions(actions)
                body.bones['torso'].ApplyLinearImpulse(impulse=self.randomImpulse, point=(0.6, 0), wake=True)
        self.randomImpulse *= 0.6

    def _evalStates(self) -> ((List[List[float]], List[Body]), List[float]):  # (state, reward)
        rewardArr = [0] * len(self.bodies)
        for i, body in enumerate(self.bodies):
            if body.active:
                bodyPos = body.getRootPos()
                if bodyPos.y >= self.MinHeight:
                    body.health = min(body.health + 1.5*self.timestep, 1)
                else:
                    body.health = max(body.health - self.timestep, 0)
                if body.health == 0:
                    body.deactivate()
                rewardArr[i] = self._calcReward(body)
                body.cumReward += rewardArr[i]
                body.maxX = max(body.maxX, bodyPos.x)
                body.timeAlive += self.timestep
        return (self._getStates(), rewardArr)

    def _calcReward(self, body: Body) -> float:
        torques = 0
        for joint in body.joints:
            torques += joint.GetMotorTorque(60) ** 2  # hardcoded fps
        return (
            2*abs(body.bones['torso'].linearVelocity.x) + body.bones['torso'].linearVelocity.x +
            + 0.1*(body.getRootPos().y > self.MinHeight)
            - torques*0.00001
            + 0.1
        )

    def _getStates(self):
        return ([a.getState() if a.active else None for a in self.bodies], [a for a in self.bodies])

    def reset(self, initState: List[float] = None) -> (List[List[float]], List[Body]):
        self.time = 0
        self.episode += 1
        off = b2Vec2(np.random.normal(0, 0.1), 0)  # same for each
        if initState is None:
            for a in self.bodies:
                a.resetState(off)
            for _ in range(4):  # make a few steps to bring joints to valid states (apply constraints)
                self.world.Step(self.timestep*2, 3, 6)
        else:
            for a in self.bodies:
                a.resetToState(initState)
        return self._getStates()

    def _recordResults(self, bodiesSorted):
        if self.record:
            self.resultRecord['Episode'].append(self.episode)
            self.resultRecord['MaxX'].append(bodiesSorted[0].maxX)
            self.resultRecord['CumulativeReward'].append(bodiesSorted[0].cumReward)
            self.resultRecord['TimeAlive'].append(bodiesSorted[0].timeAlive)
            if self.episode % 10 == 9:
                pd.DataFrame(self.resultRecord, columns=['Episode', 'MaxX', 'CumulativeReward', 'TimeAlive']).to_csv('records.csv', index=False)

    def render(self, fps=9999999) -> bool:
        return Display.clearDrawDisplay(self, fps)
