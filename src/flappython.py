import pygame
import sys
import math
import numpy as np
from random import randrange as randH
from pygame.locals import *
from nn import NeuralNetwork


POPULATION = 500

# Start pygame
pygame.init()

# Set up resolution
WINDOWX = 640  # window obj
WINDOWY = 480
windowObj = pygame.display.set_mode((WINDOWX, WINDOWY))
fpsTimer = pygame.time.Clock()
maxFPS = 30

# Ground Elevation (pixels)
groundLevel = 400

# Global colors
birdColor = pygame.Color('#222222')
backgroundColor = pygame.Color('#abcdef')
groundColor = pygame.Color('#993333')


class Pipes:

    height = 0
    width = 60
    gap = 150
    pos = 600
    replaced = False
    scored = False

    # Randomize pipe location
    def __init__(self):
        self.height = randH(210, groundLevel - 10)

    # Moves the pipes along the ground, checks if they're off the screen
    def move(self, movement):
        self.pos += movement
        if(self.pos + self.width < 0):
            return False  # Return false if we moved off the screen
        return True

    # Handles drawing the pipes to the screen
    def draw(self, surface):
        pygame.draw.rect(surface, groundColor, (self.pos,
                                                self.height, self.width, groundLevel - self.height))
        pygame.draw.rect(surface, groundColor, (self.pos, 0,
                                                self.width, self.height - self.gap))


class Bird:

    radius = 20

    def __init__(self, newPos, otherBird=None):

        self.fitness = 0
        self.score = 0
        self.velocity = 0
        self.pos = newPos

        if otherBird == None:
            self.brain = NeuralNetwork(3, 3, 1)
        else:
            self.brain = NeuralNetwork(0, 0, 0, otherBird.brain)

    # Handles drawing the bird to the screen

    def draw(self, surface):
        intPos = (int(math.floor(self.pos[0])), int(math.floor(self.pos[1])))

        pygame.draw.circle(surface, birdColor, intPos, self.radius)

    # Attempt to move the bird, make sure we aren't hitting the ground
    def move(self, movement):
        posX, posY = self.pos
        movX, movY = movement

        if((posY + movY) < groundLevel and (posY + movY) > -self.radius):
            self.pos = (posX + movX, posY + movY)
            return True  # Return if we successfuly moved

        return False

    # Test for collision with the given pipe
    def collision(self, pipe):
        posX, posY = self.pos
        collideWidth = (pipe.pos < posX + self.radius and posX -
                        self.radius < pipe.pos + pipe.width)
        collideTop = (pipe.height - pipe.gap > posY - self.radius)
        collideBottom = (posY + self.radius > pipe.height)
        if (collideWidth and (collideTop or collideBottom)):
            return True
        return False

    def up(self):
        self.velocity = -20

    def decide(self):
        pipe = findClosestPipe()
        actualy = self.pos[1] / WINDOWY
        xtopipe = (pipe.pos - self.pos[0]) / WINDOWX
        pipeheight = (pipe.height - self.pos[1]) / WINDOWY
        inputs = [actualy, xtopipe, pipeheight]
        decision = self.brain.predict(inputs)[0]
        # print(decision[0])
        if decision[0] > 0.5:
            self.up()


def initBirds():
    for i in range(0, POPULATION):
        birds.append(
            Bird((windowObj.get_width() / 4, windowObj.get_height() / 2)))


def removeBird(bird):
    savedBirds.append(bird)
    birds.remove(bird)


def normalizeFitness(birds):

    for bird in birds:
        bird.score = pow(bird.score, 2)  # 2

    sum = 0
    for bird in birds:
        sum += bird.score

    for bird in birds:
        bird.fitness = bird.score / sum


def poolSelection(birds):

    index = 0
    r = np.random.rand()

    while (r > 0):
        r -= birds[index].fitness
        index += 1

    index -= 1

    choosen = None
    max = 0
    for bird in birds:
        if bird.fitness > max:
            choosen = bird

    return choosen  # birds[index]


def newGeneration(savedBirds, birds):

    normalizeFitness(savedBirds)
    bestBird = poolSelection(savedBirds)

    for i in range(0, POPULATION):
        birds.append(
            Bird((windowObj.get_width() / 4, windowObj.get_height() / 2), bestBird))

    for i in range(1, POPULATION):
        birds[i].brain.mutate(0.1)


def resetGame():
    global pipes
    global generation
    global birds
    global savedBirds
    del pipes[:]
    pipes = [Pipes()]
    birds = []
    newGeneration(savedBirds, birds)
    savedBirds = []
    generation += 1
    print(generation)


def findClosestPipe():
    closest = None
    closestX = float("inf")
    for pipe in pipes:
        distance = pipe.pos - bird.pos[0]
        if distance > 0 and distance < closestX:
            closest = pipe
            closestX = distance

    return closest


# Setting up initial values
birds = []
savedBirds = []
initBirds()
pipes = [Pipes()]
gravity = 2
generation = 1

# Main game loop
while True:

    windowObj.fill(backgroundColor)

    # Check for events
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        # elif event.type == KEYDOWN:
        #    if (event.key == K_ESCAPE):
        #        pause()
        #    elif (event.key == K_SPACE):
        #        birds[0].up()

    # print(len(birds))
    for bird in birds:
        bird.decide()
        bird.velocity += gravity
        if (not bird.move((0, bird.velocity))):
            bird.pos = bird.pos
        bird.score += 1

    for pipe in pipes:

        if not pipe.replaced and pipe.pos < windowObj.get_width() / 2:
            pipes[len(pipes):] = [Pipes()]
            pipe.replaced = True

        pipe.draw(windowObj)

        for bird in birds:
            if (bird.collision(pipe)):
                removeBird(bird)

        if (not pipe.move(-10)):
            del pipe

    if len(birds) == 0:
        resetGame()

    # if (generation < 500):
    #    continue

    pygame.draw.rect(windowObj, groundColor, (0, groundLevel,
                                              windowObj.get_width(), windowObj.get_height()))

    for bird in birds:
        bird.draw(windowObj)

    pygame.display.update()
    fpsTimer.tick(maxFPS)
