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


def CreateBone(world, pos=(0, 5), angle=-b2_pi*0.5, scale=(1, 1), anchor0=-0.95, anchor1=0.95):
    size = (1*scale[0], 0.2*scale[1])
    bone = world.CreateDynamicBody(position=pos,
                                   angle=angle,
                                   allowSleep=False,
                                   fixtures=b2FixtureDef(density=1.0,
                                                         friction=0.6,
                                                         shape=b2PolygonShape(box=size),
                                                         categoryBits=0x0004, maskBits=0x0002))
    bone.ms_joints = []
    bone.ms_anchor = [(anchor0 * size[0], 0), (anchor1 * size[0], 0)]
    return bone


def ConnectBones(world, mainBone, otherBone, anchorMain=1, anchorOther=0, angleLow=-b2_pi*0.25, angleHigh=b2_pi*0.25, maxTorque=10):
    joint = world.CreateRevoluteJoint(bodyA=mainBone,
                                      bodyB=otherBone,
                                      localAnchorA=mainBone.ms_anchor[anchorMain],  # przestrzeń lokalna
                                      localAnchorB=otherBone.ms_anchor[anchorOther],
                                      lowerAngle=angleLow,  # względem ciała A
                                      upperAngle=angleHigh,
                                      enableLimit=True,
                                      maxMotorTorque=maxTorque,
                                      motorSpeed=0.0,  # prędkość kątowa
                                      enableMotor=False)
    mainBone.ms_joints.append(joint)
    return joint


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption('Simple pygame example')
    clock = pygame.time.Clock()

    b2World.CreateBone = CreateBone
    b2World.ConnectBones = ConnectBones
    world = b2World(gravity=(0, -10), doSleep=True)

    ground_body = world.CreateStaticBody(
        fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-1e5, 0), (1e5, 0)]),
                              categoryBits=0x0002, maskBits=0x0004),
        position=(0, 0)
    )

    torso = world.CreateBone(scale=(1.4, 1.6))
    thigh1 = world.CreateBone(pos=(0, 4))
    thigh2 = world.CreateBone(pos=(0, 4))
    crus1 = world.CreateBone(pos=(0, 3), scale=(0.8, 0.8))
    crus2 = world.CreateBone(pos=(0, 3), scale=(0.8, 0.8))
    foot1 = world.CreateBone(pos=(0, 2), scale=(0.4, 0.5), angle=0)
    foot2 = world.CreateBone(pos=(0, 2), scale=(0.4, 0.5), angle=0)

    world.ConnectBones(torso, thigh1, angleLow=-b2_pi*0.5, angleHigh=b2_pi*0.5)
    world.ConnectBones(torso, thigh2, angleLow=-b2_pi*0.5, angleHigh=b2_pi*0.5)
    world.ConnectBones(thigh1, crus1, angleLow=-b2_pi*0.9, angleHigh=0)
    world.ConnectBones(thigh2, crus2, angleLow=-b2_pi*0.9, angleHigh=0)
    world.ConnectBones(crus1, foot1)
    world.ConnectBones(crus2, foot2)

    # world.ConnectBones(world.CreateBone(), world.CreateBone(angle=b2_pi*0.5), 1, 1)
    # world.ConnectBones(world.CreateBone(), world.CreateBone(angle=b2_pi*0.5), 1, 1)

    # bone1 = world.CreateDynamicBody(position=(-1.5, 1), angle=0, allowSleep=False,
    #                                 fixtures=b2FixtureDef(density=1.0, friction=0.6,
    #                                                       shape=b2PolygonShape(box=(1.2, 0.2))))
    # bone2 = world.CreateDynamicBody(position=(1.5, 1), angle=0, allowSleep=False,
    #                                 fixtures=b2FixtureDef(density=1.0, friction=0.6,
    #                                                       shape=b2PolygonShape(box=(1.2, 0.2))))
    # print(type(bone1))
    # joint = world.CreateRevoluteJoint(bodyA=bone1,
    #                                   bodyB=bone2,
    #                                   localAnchorA=(1.2, 0.0),  # przestrzeń lokalna
    #                                   localAnchorB=(-1.2, 0.0),
    #                                   lowerAngle=-0.25 * b2_pi,  # względem ciała A
    #                                   upperAngle=0.25 * b2_pi,
    #                                   enableLimit=True,
    #                                   maxMotorTorque=500.0,
    #                                   motorSpeed=0.0,  # prędkość kątowa
    #                                   enableMotor=True)

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
            # joint.motorSpeed = (random.random() - 0.5) * 10
            time = 0
        world.Step(TIME_STEP, 10, 10)

        # Flip the screen and try to keep at the target FPS
        pygame.display.flip()
        clock.tick(TARGET_FPS)
        # print(joint.motorSpeed)

    pygame.quit()
    print('Done!')


if __name__ == "__main__":
    main()
