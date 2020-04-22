from Actor import Actor
from Box2D.Box2D import (b2World, b2Body, b2FixtureDef, b2EdgeShape, b2_pi)
import numpy as np
from Neural import Agent, makeModel
import pandas as pd
from os.path import isfile
import gc

MinHeight = 0.6

NumberOfActors = 100
SFA = 1 #Skip first actor
IA = 5 #Important actors #nie mam pomyslu na nazwe xd

class Environment:
    def __init__(self, baseModelPath=""):
        self.world = b2World(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
                                  categoryBits=0x0002, maskBits=0x0004),
            position=(0, 0),
            userData = 'ground'
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

        self.actors = [Actor(self.world) for i in range(0, NumberOfActors)]
        
        if baseModelPath is not "":
            baseModel = makeModel()
            baseModel.load_weights(baseModelPath)
            self.actors[0].reset(0, 0, baseModel)
            self.actors[0].deactivate()
            for actor in self.actors[SFA:]:
                actor.reset(0.9, 0.1, baseModel)
        self.time = 0
        self.gen = 0
        self.genTime = 50
        self.tick = 0

    def step(self, timestep, record=False):
        self.time += timestep
        self.world.Step(timestep, 6, 4)
        # if self.tick == 0:
        for actor in self.actors[SFA:]:
            if actor.active:
                actor.applyOutputArray(actor.model.predict(actor.getInputArray())[0])

        for actor in self.actors[SFA:]:
            if actor.active:
                actorPos = actor.getRootPos()
                actor.reward += actor.calculateReward()
                actor.prevPos = actorPos.x
                actor.timeAlive += timestep
                actor.maxX = max(actor.maxX, actorPos.x)
            #     print(actor.reward)
        # print()
        for actor in self.actors[SFA:]:
            if actor.reward < -20:
                actor.deactivate()
                continue

        self.tick = (self.tick + 1) % 3

        resetNow = True
        for actor in self.actors[SFA:]:
            if actor.active:
                resetNow = False

        if self.time >= self.genTime or resetNow:
            self.actors.sort(key=lambda act: act.maxX, reverse=True)
            if record:
                for idx, actor in enumerate(self.actors[:IA]):
                    self.resultRecord['Gen'].append(self.gen)
                    self.resultRecord['Idx'].append(idx)
                    self.resultRecord['MaxX'].append(actor.maxX)
                    self.resultRecord['EndReward'].append(actor.reward)
                    self.resultRecord['TimeAlive'].append(actor.timeAlive)
                if self.gen % 10 == 9:
                    pd.DataFrame(self.resultRecord, columns=['Gen', 'Idx', 'MaxX', 'EndReward', 'TimeAlive']).to_csv('records.csv', index=False)

            print("!!!!!!!!!!!!!!!!!!!!! ", self.gen, "Max x: %.2f" % self.actors[0].maxX, " !!!!!!!!!!!!!!!!!!!!!")

            for actor in self.actors[IA:]:
                actor.reset(0.9, 0.5, self.actors[int(np.random.uniform(0, IA))].model)
            
            self.actors[0].model.save('models/Gen%d_%d' % (self.gen, 0))
            self.actors[0].maxX = 0
            for idx, actor in enumerate(self.actors[1:IA]):
                actor.model.save('models/Gen%d_%d' % (self.gen, idx+1))
                actor.reset(0.9, 0.1)

            self.gen += 1
            self.time = 0
            self.tick = 0

    def getActors(self):
        return self.actors

class Environment_DDPG:
    def __init__(self, loadLastCheckpoint = False):
        self.world = b2World(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
                                  categoryBits=0x0002, maskBits=0x0004),
            position=(0, 0),
            userData = 'ground'
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
                             'MaxX': [],
                             'EndReward': [],
                             'TimeAlive': []}

        self.actor = Actor(self.world, noModel=True)
        self.agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[28], tau=0.001,
                     batch_size=64, layer1_size=800, layer2_size=600, n_actions=6)
        if loadLastCheckpoint:
            self.agent.load_models()
        
        self.time = 0
        self.gen = 0
        self.genTime = 50
        self.tick = 0
        self.observation = self.actor.getInputArray()
        self.maxX = 0

    def step(self, timestep, record=False):
        done = False
        self.time += timestep
        self.world.Step(timestep, 6, 4)
        # if self.tick == 0:
        action = self.agent.choose_action(self.observation)
        self.actor.applyOutputArray(action)

        reward = self.actor.calculateReward()
        self.actor.reward += reward
        actorPos = self.actor.getRootPos()
        self.actor.prevPos = actorPos.x
        self.actor.timeAlive += timestep
        self.actor.maxX = max(self.actor.maxX, actorPos.x)
        
        if self.actor.reward < -20:
            self.actor.deactivate()
            done = True

        self.tick = (self.tick + 1) % 3

        new_state = self.actor.getInputArray()
        self.agent.remember(self.observation, action, reward, new_state, int(done))
        self.agent.learn()
        self.observation = new_state

        if self.time >= self.genTime or done:
            if record:
                self.resultRecord['Gen'].append(self.gen)
                self.resultRecord['MaxX'].append(self.actor.maxX)
                self.resultRecord['EndReward'].append(self.actor.reward)
                self.resultRecord['TimeAlive'].append(self.actor.timeAlive)
                if self.gen % 10 == 9:
                    pd.DataFrame(self.resultRecord, columns=['Gen', 'MaxX', 'EndReward', 'TimeAlive']).to_csv('records_ddpg.csv', index=False)
            
            if self.maxX < self.actor.maxX:
                self.maxX = self.actor.maxX

            print("!!!!!!!!!!!!!!!!!!!!! ", self.gen, "Max x: %.2f, All time max x: %.2f" % (self.actor.maxX, self.maxX), " !!!!!!!!!!!!!!!!!!!!!")

            self.actor.softReset()
            self.observation = self.actor.getInputArray()

            if self.gen % 30 == 29:
                self.agent.save_models()

            self.gen += 1
            self.time = 0
            self.tick = 0
    
    def getActors(self):
        actors = []
        actors.append(self.actor)
        return actors