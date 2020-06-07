from Box2D.Box2D import (b2World, b2Body, b2FixtureDef, b2EdgeShape, b2_pi, b2PolygonShape)
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import pygame

PPM = 50.0  # pixels per meter
TARGET_FPS = 99999
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
screenOffset = (SCREEN_WIDTH * 0.1, SCREEN_HEIGHT * 0.9)


def my_draw_polygon(polygon, transform, color, screen):
    vertices = [(transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0] + screenOffset[0], -v[1] + screenOffset[1]) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)


def my_draw_edge(edge, transform, color, screen):
    vertices = [(transform * v) * PPM for v in edge.vertices]
    vertices = [(v[0] + screenOffset[0], -v[1] + screenOffset[1]) for v in vertices]
    pygame.draw.line(screen, color, vertices[0], vertices[1])


def drawBody(body, screen, color=(255, 255, 255, 255)):
    for fixture in body.fixtures:
        fixture.shape.draw(body.transform, color, screen)


b2Body.draw = drawBody
b2PolygonShape.draw = my_draw_polygon
b2EdgeShape.draw = my_draw_edge

screen = None
display = 1
clock = pygame.time.Clock()


def clearDrawDisplay(env, fps):
    global screen
    global display
    global screenOffset
    if screen is None:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    running = True
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                display = (display + 1) % 3
            if pygame.K_1 <= event.key <= pygame.K_9:
                env.episodeTime = (event.key - pygame.K_1 + 1) ** 2
                print("Episode time = ", env.episodeTime)
    env.helperEdge1.position.x = (int((env.bodies[0].getRootPos().x + 1) / 5) + 1) * 5
    clock.tick(fps if display != 0 else 99999999)
    pygame.display.set_caption('Muscular System 2D, alive = %d, fps = %.2f' % (len([0 for a in env.bodies if a.active]), clock.get_fps()))
    if display != 0:
        screen.fill((0, 0, 0, 0))

        env.ground_body.draw(screen)
        env.helperEdge1.draw(screen)
        env.helperEdge2.draw(screen)

        if display == 1:
            sBodies = sorted(env.bodies, key=lambda ag: ag.cumReward, reverse=True)
            for body in sBodies[:5]:
                if body.active:
                    body.draw(screen)
            screenOffset = (SCREEN_WIDTH * 0.5 - sBodies[0].getRootPos().x * PPM, SCREEN_HEIGHT * 0.9)
        else:
            for body in env.bodies:
                if body.active:
                    body.draw(screen)
        pygame.display.flip()
    return running
