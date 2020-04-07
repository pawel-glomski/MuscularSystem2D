import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
from pygame import Vector2 as Vec2

import numpy as np
import random
from Box2D import *
from Environment import Environment

import Neural


PPM = 35.0  # pixels per meter
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
    pygame.display.set_caption('Muscular System 2D')
    clock = pygame.time.Clock()

    env = Environment()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        env.step(TIME_STEP)

        screen.fill((0, 0, 0, 0))

        env.ground_body.draw(screen)
        for actor in env.actors:
            actor.draw(screen)

        pygame.display.flip()

        if clock.get_time() != 0:
            print(1 / (clock.get_time()/1000))
        clock.tick(TARGET_FPS)

    pygame.quit()
    print('Done!')


if __name__ == "__main__":
    main()
