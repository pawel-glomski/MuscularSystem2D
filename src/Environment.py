from Actor import Actor
from Box2D.Box2D import (b2World, b2Body, b2FixtureDef, b2EdgeShape, b2_pi)
import numpy as np
import Neural

MinHeight = 0.6


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

        self.actors = [Actor(self.world) for i in range(0, 100)]
        self.time = 0
        self.gen = 0
        self.genTime = 18
        self.tick = 0

    def step(self, timestep):
        self.time += timestep
        if self.tick == 0:
            for actor in self.actors:
                if actor.reward < -20:
                    actor.deactivate()
                    continue
                actor.applyOutputArray(actor.model.predict(actor.getInputArray()))
        self.world.Step(timestep, 6, 4)

        for actor in self.actors:
            if actor.active:
                actorPos = actor.getRootPos()
                actor.reward += 10*(actorPos.x - actor.prevPos) - \
                    abs(actor.bones['torso'].angle + b2_pi*0.5) * 0.1 - 0.5 * max(0, MinHeight - actorPos.y)
                actor.prevPos = actorPos.x
                actor.timeAlive += timestep
            #     print(actor.reward)
        # print()

        self.tick = (self.tick + 1) % 2

        resetNow = True
        for actor in self.actors:
            if actor.active:
                resetNow = False

        if self.time >= self.genTime or resetNow:
            self.actors.sort(key=lambda act: act.reward + act.timeAlive, reverse=True)
            if self.actors[0].reward <= -10 and self.actors[0].timeAlive < 1.0:
                for actor in self.actors:
                    actor.reset(0.9, 1)
            else:
                for actor in self.actors[5:]:
                    actor.reset(0.9, 0.5, self.actors[int(np.random.uniform(0, 5))].model)

                self.actors[0].model.save('models/Gen%d_%d' % (self.gen, 0))
                self.actors[0].reset(0, 0)
                for idx, actor in enumerate(self.actors[1:5]):
                    actor.model.save('models/Gen%d_%d' % (self.gen, idx+1))
                    actor.reset(0.9, 0.1)
                self.actors[0].reset(0, 0)

            self.gen += 1
            print("!!!!!!!!!!!!!!!!!!!!! ", self.gen,  " !!!!!!!!!!!!!!!!!!!!!")
            self.time = 0
            self.tick = 0
