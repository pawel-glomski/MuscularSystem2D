from Actor import Actor
from Box2D.Box2D import (b2World, b2Body, b2FixtureDef, b2EdgeShape)
import numpy as np
import Neural


class Environment:
    def __init__(self):
        self.world = b2World(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
                                  categoryBits=0x0002, maskBits=0x0004),
            position=(0, 0)
        )
        self.actors = [Actor(self.world) for i in range(0, 100)]

    def step(self, timestep):
        for actor in self.actors:
            actor.applyOutputArray(actor.model.predict(actor.getInputArray()))

        self.world.Step(timestep, 10, 10)
