from pygame import Vector2 as Vec2
import numpy as np
from Box2D import *
import Utils
import copy
import keras

RadToDeg = 180/b2_pi
StartTransf = {}


class Body:
    def __init__(self, world: b2World):
        self.joints = []
        self.bones = {}
        self.secondColor = (155, 155, 155, 255)
        self.addBone(world, 'torso', size=(0.6, 0.2), angle=-b2_pi*0.5, density=275)
        self.addBone(world, 'thigh1', 'torso', angleLow=-b2_pi*0.4, angleHigh=b2_pi * 0.4, size=(0.4, 0.15), angle=-b2_pi*0.5, color=self.secondColor)
        self.addBone(world, 'thigh2', 'torso', angleLow=-b2_pi*0.4, angleHigh=b2_pi * 0.4, size=(0.4, 0.15), angle=-b2_pi*0.5)
        self.addBone(world, 'crus1',  'thigh1', angleLow=-b2_pi*0.9, angleHigh=0, size=(0.35, 0.1), angle=-b2_pi*0.5, color=self.secondColor)
        self.addBone(world, 'crus2',  'thigh2', angleLow=-b2_pi*0.9, angleHigh=0, size=(0.35, 0.1), angle=-b2_pi*0.5)
        self.addBone(world, 'foot1',  'crus1', angleLow=-b2_pi*0.3, angleHigh=b2_pi*0.2, size=(0.275, 0.075), color=self.secondColor)
        self.addBone(world, 'foot2',  'crus2', angleLow=-b2_pi*0.3, angleHigh=b2_pi*0.2, size=(0.275, 0.075))
        self.active = True
        self.cumReward = 0
        self.timeAlive = 0
        self.maxX = 0
        self.health = 1
        self.resetState()

    def getState(self):
        states = []
        rootBone = self.bones['torso']
        rootPos = self.getRootPos()

        states.append(self.health)
        states.append(rootPos.y)
        states.append(rootBone.angle + b2_pi/2)
        states.append(rootBone.angularVelocity)
        states += rootBone.linearVelocity
        for bone in list(self.bones.values())[1:]:  # pomijamy roota
            states += bone.position - rootPos
            states += bone.linearVelocity - rootBone.linearVelocity
        return np.reshape(np.array(states), (1, len(states)))

    def applyActions(self, actions):
        for joint, torque in zip(self.joints, actions):
            joint.maxMotorTorque = abs(float(torque)) * 75
            joint.motorSpeed = np.copysign(9999999, float(torque))

    def draw(self, screen):
        bonesList = list(self.bones.values())
        for bone in [bone for bone in bonesList if bone.color == self.secondColor]:
            bone.draw(screen, bone.color)
        for bone in [bone for bone in bonesList if bone.color != self.secondColor]:
            bone.draw(screen, bone.color)

    def destroy(self, world: b2World):
        for joint in self.joints:
            world.DestroyJoint(joint)
        for bone in self.bones.values():
            world.DestroyBody(bone)

    def addBone(self, world, name, parentName='', size=(1, 0.2), angle=0, pos=(0, 1.25), anchor0=-0.95, anchor1=0.95, parentAnchor=1, thisAnchor=0,
                angleLow=0, angleHigh=0, maxTorque=400, color=(255, 255, 255, 255), density=175.0):
        size = (size[0] / 2, size[1] / 2)
        parent = None if parentName == '' else self.bones[parentName]
        # angle = angle if parent is None else parent.angle
        pos = pos if parent is None else (parent.position + Vec2(parent.ms_anchor[parentAnchor][0] + size[0], 0).rotate(angle*RadToDeg))

        StartTransf[name] = (pos, angle)

        bone = world.CreateDynamicBody(position=pos,
                                       angle=angle,
                                       allowSleep=False,
                                       fixtures=b2FixtureDef(density=density,
                                                             friction=9.0,
                                                             shape=b2PolygonShape(box=size),
                                                             categoryBits=0x0004, maskBits=0x0002))
        bone.color = color
        bone.ms_anchor = [Vec2(anchor0 * size[0], 0),
                          Vec2(anchor1 * size[0], 0)]

        self.bones[name] = bone
        if parent is not None:
            self.joints.append(world.CreateRevoluteJoint(bodyA=parent,
                                                         bodyB=bone,
                                                         # przestrzeń lokalna
                                                         localAnchorA=parent.ms_anchor[parentAnchor],
                                                         localAnchorB=bone.ms_anchor[thisAnchor],
                                                         lowerAngle=angleLow,  # względem ciała A
                                                         upperAngle=angleHigh,
                                                         enableLimit=True,
                                                         maxMotorTorque=maxTorque,
                                                         motorSpeed=0.0,  # prędkość kątowa
                                                         enableMotor=True,
                                                         collideConnected=False))

    def getRootPos(self):
        return self.bones['torso'].position

    def deactivate(self):
        self.active = False
        for bone in self.bones.values():
            bone.active = False

    def resetState(self, legsOffset=b2Vec2(0, 0)):
        self.health = 1
        self.cumReward = 0
        self.timeAlive = 0
        self.active = True
        self.maxX = -float('inf')
        for joint in self.joints:
            joint.motorSpeed = 0
            joint.maxMotorTorque = 0
        for name, bone in self.bones.items():
            bone.position = StartTransf[name][0]
            bone.angle = StartTransf[name][1]
            bone.linearVelocity = b2Vec2(0, 0)
            bone.angularVelocity = 0
            bone.active = True

        self.bones['thigh1'].position += legsOffset
        self.bones['crus1'].position += 2*legsOffset
        self.bones['foot1'].position += 2*legsOffset
        self.bones['thigh2'].position -= legsOffset
        self.bones['crus2'].position -= 2*legsOffset
        self.bones['foot2'].position -= 2*legsOffset

    def getRealStates(self):
        state = [float(self.health)]
        for bone in self.bones.values():
            state.append(b2Vec2(bone.position))
            state.append(float(bone.angle))
            state.append(b2Vec2(bone.linearVelocity))
            state.append(float(bone.angularVelocity))
        return state

    def resetToState(self, state):

        def postInc():
            postInc.i += 1
            return postInc.i - 1
        postInc.i = 0

        self.health = float(state[postInc()])
        self.cumReward = 0
        self.timeAlive = 0
        self.active = True
        self.maxX = -float('inf')
        for bone in self.bones.values():
            bone.position = b2Vec2(state[postInc()])
            bone.angle = float(state[postInc()])
            bone.linearVelocity = b2Vec2(state[postInc()])
            bone.angularVelocity = float(state[postInc()])
            bone.active = True
