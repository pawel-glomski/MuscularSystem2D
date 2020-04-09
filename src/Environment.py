from Actor import Actor
from Box2D.Box2D import (b2World, b2Body, b2FixtureDef, b2EdgeShape, b2_pi)
import numpy as np
import Neural
import pandas as pd
from os.path import isfile
import gc

MinHeight = 0.6


class Environment:
    def __init__(self, baseModelPath=""):
        self.world = b2World(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
                                  categoryBits=0x0002, maskBits=0x0004),
            position=(0, 0)
        )
        self.helperEdge1 = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(0, -1e5), (0, 1e5)]),
                                  categoryBits=0x0004, maskBits=0x0002),
            position=(0, 0)
        )
        self.helperEdge2 = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, MinHeight), (1e5, MinHeight)]),
                                  categoryBits=0x0004, maskBits=0x0002),
            position=(0, 0)
        )

        self.resultRecord = {'Gen': [],
                             'Idx': [],
                             'MaxX': [],
                             'EndReward': [],
                             'TimeAlive': []}
        self.actors = [Actor(self.world) for i in range(0, 100)]
        if baseModelPath is not "":
            baseModel = Neural.makeModel()
            baseModel.load_weights(baseModelPath)
            self.actors[0].reset(0, 0, baseModel)
            self.actors[0].deactivate()
            for actor in self.actors[1:]:
                actor.reset(0.9, 0.1, baseModel)
        self.time = 0
        self.gen = 0
        self.genTime = 50
        self.tick = 0

    def step(self, timestep, record=False):
        self.time += timestep
        self.world.Step(timestep, 6, 4)
        # if self.tick == 0:
        for actor in self.actors[1:]:
            if actor.reward < -20:
                actor.deactivate()
                continue
            actor.applyOutputArray(actor.model.predict(actor.getInputArray()))

        for actor in self.actors[1:]:
            if actor.active:
                actorPos = actor.getRootPos()
                actor.reward += 10*(actorPos.x - actor.prevPos) - \
                    abs(actor.bones['torso'].angle + b2_pi*0.5) * 0.5 - 0.5 * max(0, MinHeight - actorPos.y)
                actor.prevPos = actorPos.x
                actor.timeAlive += timestep

                actor.maxX = max(actor.maxX, actorPos.x)
            #     print(actor.reward)
        # print()

        self.tick = (self.tick + 1) % 3

        resetNow = True
        for actor in self.actors[1:]:
            if actor.active:
                resetNow = False

        if self.time >= self.genTime or resetNow:
            self.actors.sort(key=lambda act: act.maxX, reverse=True)
            if record:
                for idx, actor in enumerate(self.actors[:5]):
                    self.resultRecord['Gen'].append(self.gen)
                    self.resultRecord['Idx'].append(idx)
                    self.resultRecord['MaxX'].append(actor.maxX)
                    self.resultRecord['EndReward'].append(actor.reward)
                    self.resultRecord['TimeAlive'].append(actor.timeAlive)
                if self.gen % 10 == 9:
                    pd.DataFrame(self.resultRecord, columns=['Gen', 'Idx', 'MaxX', 'EndReward', 'TimeAlive'], index=False).to_csv('records.csv')

            print("!!!!!!!!!!!!!!!!!!!!! ", self.gen, "Max x: %.2f" % self.actors[0].maxX, " !!!!!!!!!!!!!!!!!!!!!")

            for actor in self.actors[5:]:
                actor.reset(0.9, 0.5, self.actors[int(np.random.uniform(0, 5))].model)

            self.actors[0].model.save('models/Gen%d_%d' % (self.gen, 0))
            for idx, actor in enumerate(self.actors[1:5]):
                actor.model.save('models/Gen%d_%d' % (self.gen, idx+1))
                actor.reset(0.9, 0.1)

            self.gen += 1
            self.time = 0
            self.tick = 0
