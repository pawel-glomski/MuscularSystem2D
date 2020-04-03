import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
from pygame import Vector2 as Vec2

import numpy as np
import random

import Box2D
from Box2D import *

from actor import *

PPM = 35.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
COLORS = {
    b2_staticBody: (255, 255, 255, 255),
    b2_dynamicBody: (127, 127, 127, 255),
}


def my_draw_polygon(polygon, body, fixture, screen):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0] + SCREEN_WIDTH / 2, -v[1] + SCREEN_HEIGHT / 2) for v in vertices]
    pygame.draw.polygon(screen, COLORS[body.type], vertices)


def my_draw_edge(edge, body, fixture, screen):
    vertices = [(body.transform * v) * PPM for v in edge.vertices]
    vertices = [(SCREEN_WIDTH / 2 - v[0], SCREEN_HEIGHT / 2 - v[1]) for v in vertices]
    pygame.draw.line(screen, COLORS[body.type], vertices[0], vertices[1])


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption('Muscular System 2D')
    clock = pygame.time.Clock()

    b2Body.ms_joints = []
    b2World.CreateBone = CreateBone
    b2World.CreateActor = Actor.CreateActor
    world = b2World(gravity=(0, -10), doSleep=True)

    ground_body = world.CreateStaticBody(
        fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
                              categoryBits=0x0002, maskBits=0x0004),
        position=(0, 0)
    )
    actor = world.CreateActor()

    b2PolygonShape.draw = my_draw_polygon
    b2EdgeShape.draw = my_draw_edge
    time = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
        i = 1
        for bone in actor.bones.values():
            for joint in bone.ms_joints:
                joint.motorSpeed = i*np.sin(time)
                i *= -1

        world.Step(TIME_STEP, 10, 10)
        time += TIME_STEP
        actor.getInputArray()
        screen.fill((0, 0, 0, 0))
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture, screen)
        pygame.display.flip()
        clock.tick(TARGET_FPS)
        # print(joint.motorSpeed)

    pygame.quit()
    print('Done!')


if __name__ == "__main__":
    main()
