import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
from pygame import Vector2 as Vec2

import numpy as np
import random
from Box2D import *
from Environment import Environment

import Neural


PPM = 50.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600


def my_draw_polygon(polygon, transform, color, screen):
    vertices = [(transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0] + SCREEN_WIDTH / 2, -v[1] + SCREEN_HEIGHT / 2) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)


def my_draw_edge(edge, transform, color, screen):
    vertices = [(transform * v) * PPM for v in edge.vertices]
    vertices = [(SCREEN_WIDTH / 2 - v[0], SCREEN_HEIGHT / 2 - v[1]) for v in vertices]
    pygame.draw.line(screen, color, vertices[0], vertices[1])


def drawBody(body, screen, color=(255, 255, 255, 255)):
    for fixture in body.fixtures:
        fixture.shape.draw(body.transform, color, screen)


def main():
    b2Body.draw = drawBody
    b2PolygonShape.draw = my_draw_polygon
    b2EdgeShape.draw = my_draw_edge

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    clock = pygame.time.Clock()

    env = Environment('models/Gen4_0')

    running = True
    display = 1
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    display = (display + 1) % 3
                if pygame.K_1 <= event.key <= pygame.K_9:
                    env.genTime = (event.key - pygame.K_1 + 1) ** 2
                    print("Gen time = ", env.genTime)

        env.step(TIME_STEP, True)

        if display != 0:
            screen.fill((0, 0, 0, 0))
            env.ground_body.draw(screen)
            env.helperEdge1.draw(screen)
            env.helperEdge2.draw(screen)

            if display == 1:
                for actor in env.actors[:5]:
                    if actor.active:
                        actor.draw(screen)
                pygame.display.set_caption("Pos: %.2f" % env.actors[0].getRootPos().x + ", Reward: %.2f" % env.actors[0].reward)
            else:
                for actor in env.actors:
                    if actor.active:
                        actor.draw(screen)
            pygame.display.flip()

        if clock.get_time() != 0 and display != 1:
            pygame.display.set_caption('Muscular System 2D %.2f' % clock.get_fps())
        clock.tick(9999)

    pygame.quit()
    print('Done!')


if __name__ == "__main__":
    main()
