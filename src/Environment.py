from Actor import Actor
from Box2D.Box2D import (b2World, b2Body, b2FixtureDef, b2EdgeShape, b2_pi)
import numpy as np
import Neural

MinHeight = 0.5


class Environment:
    def __init__(self):
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

        self.actors = [Actor(self.world) for i in range(0, 50)]
        self.time = 0
        self.ignoreFirstTicks = True
        self.gen = 0
        self.genTime = 2

    def step(self, timestep):
        self.time += timestep

        for actor in self.actors:
            actor.applyOutputArray(actor.model.predict(actor.getInputArray()))
        self.world.Step(timestep, 10, 10)

        if self.time > 0.08 and self.ignoreFirstTicks:
            self.ignoreFirstTicks = False

        if not self.ignoreFirstTicks:
            for actor in self.actors:
                actorPos = actor.getRootPos()
                actor.reward += 1*(actorPos.x - actor.prevPos) - abs(actor.bones['torso'].angle + b2_pi*0.5) * 0.1 + min(0, actorPos.y - MinHeight)
                actor.prevPos = actorPos.x
            #     print(actor.reward)
            # print()

        if self.time >= self.genTime:
            self.actors.sort(key=lambda act: act.reward, reverse=True)
            self.actors[0].reset(0, 0)
            for actor in self.actors[5:]:
                actor.reset(0.5, 1, self.actors[int(np.random.uniform(0, 5))].model)
            for actor in self.actors[1:5]:
                actor.reset(0.5, 2)

            self.gen += 1
            print("!!!!!!!!!!!!!!!!!!!!! ", self.gen,  " !!!!!!!!!!!!!!!!!!!!!")
            self.time = 0
            self.ignoreFirstTicks = True
