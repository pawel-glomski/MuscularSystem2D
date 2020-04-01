import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

import numpy as np
import random

import Box2D
from Box2D import *

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
    pygame.display.set_caption('Simple pygame example')
    clock = pygame.time.Clock()

    world = b2World(gravity=(0, -10), doSleep=True)

    ground_body = world.CreateStaticBody(
        shapes=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
        position=(0, 0)
    )

    bone1 = world.CreateDynamicBody(position=(-1.5, 1), angle=0, allowSleep=False,
                                    fixtures=b2FixtureDef(density=1.0, friction=0.6,
                                                          shape=b2PolygonShape(box=(1.2, 0.2))))
    bone2 = world.CreateDynamicBody(position=(1.5, 1), angle=0, allowSleep=False,
                                    fixtures=b2FixtureDef(density=1.0, friction=0.6,
                                                          shape=b2PolygonShape(box=(1.2, 0.2))))
    joint = world.CreateRevoluteJoint(bodyA=bone1,
                                      bodyB=bone2,
                                      localAnchorA=(1.2, 0.0),  # przestrzeń lokalna
                                      localAnchorB=(-1.2, 0.0),
                                      lowerAngle=-0.25 * b2_pi,  # względem ciała A
                                      upperAngle=0.25 * b2_pi,
                                      enableLimit=True,
                                      maxMotorTorque=500.0,
                                      motorSpeed=0.0,  # prędkość kątowa
                                      enableMotor=True)

    b2PolygonShape.draw = my_draw_polygon
    b2EdgeShape.draw = my_draw_edge
    time = 0
    running = True
    while running:

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        screen.fill((0, 0, 0, 0))
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture, screen)

        time += TIME_STEP
        if time >= 0.2:
            joint.motorSpeed = (random.random() - 0.5) * 10
            time = 0
        world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        clock.tick(TARGET_FPS)
        print(joint.motorSpeed)

    pygame.quit()
    print('Done!')


if __name__ == "__main__":
    main()
