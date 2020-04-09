from pygame import Vector2 as Vec2
import numpy as np
from Box2D import *
import Neural
import copy

RadToDeg = 180/b2_pi
StartTransf = {}


class Actor:
    def __init__(self, world: b2World, modelPath=None):
        self.joints = []
        self.bones = {}
        self.addBone(world, 'torso', size=(0.6, 0.2), angle=-b2_pi*0.5)

        self.addBone(world, 'thigh1', 'torso',  angleLow=-b2_pi*0.25,   angleHigh=b2_pi*0.5,    size=(0.4, 0.15), color=(155, 155, 155, 255))
        self.addBone(world, 'crus1',  'thigh1', angleLow=-b2_pi*0.9,    angleHigh=0,            size=(0.35, 0.1), color=(155, 155, 155, 255))
        self.addBone(world, 'foot1',  'crus1',  angleLow=0,             angleHigh=b2_pi*0.65,   size=(0.25, 0.05), color=(155, 155, 155, 255))

        self.addBone(world, 'thigh2', 'torso',  angleLow=-b2_pi*0.25, angleHigh=b2_pi*0.75,    size=(0.4, 0.15))
        self.addBone(world, 'crus2',  'thigh2', angleLow=-b2_pi*0.9,  angleHigh=0,            size=(0.35, 0.1))
        self.addBone(world, 'foot2',  'crus2',  angleLow=0,           angleHigh=b2_pi*0.65,   size=(0.2, 0.05))

        self.model = Neural.makeModel()
        if modelPath is not None:
            self.model.load_weights(modelPath)

        self.reward = 0
        self.prevPos = 0
        self.active = True
        self.timeAlive = 0
        self.maxX = 0
        self.reset(0, 0)

    def getInputArray(self):
        inputs = []
        rootBone = self.bones['torso']
        rootPos = self.getRootPos()

        inputs.append(rootPos.y)
        inputs.append(rootBone.angle + b2_pi*0.5)
        inputs += rootBone.linearVelocity + Vec2(0, rootBone.ms_anchor[1][0] * rootBone.angularVelocity).rotate(rootBone.angle * RadToDeg)
        for bone in list(self.bones.values())[1:]:  # pomijamy roota
            inputs += (bone.position + Vec2(bone.ms_anchor[1]).rotate(bone.angle * RadToDeg)) - rootPos
            inputs += bone.linearVelocity + Vec2(0, bone.ms_anchor[1][0] * bone.angularVelocity).rotate(bone.angle * RadToDeg)
        return np.reshape(np.array(inputs), (1, len(inputs)))

    def applyOutputArray(self, outputArr):
        # print(outputArr)
        for joint, speed in zip(self.joints, outputArr[0]):
            joint.motorSpeed = (float(speed) - 0.5) * 50

    def draw(self, screen):
        bonesList = list(self.bones.values())
        for bone in bonesList[1:]:
            bone.draw(screen, bone.color)
        bonesList[0].draw(screen)

    def destroy(self, world: b2World):
        for joint in self.joints:
            world.DestroyJoint(joint)
        for bone in self.bones.values():
            world.DestroyBody(bone)

    def addBone(self, world, name, parentName='', size=(1, 0.2), angle=0, pos=(0, 1.5), anchor0=-0.95, anchor1=0.95, parentAnchor=1, thisAnchor=0, angleLow=0, angleHigh=0, maxTorque=400, color=(255, 255, 255, 255)):
        size = (size[0] / 2, size[1] / 2)
        parent = None if parentName == '' else self.bones[parentName]
        angle = angle if parent is None else parent.angle
        pos = pos if parent is None else (parent.position + Vec2(parent.ms_anchor[parentAnchor][0] + size[0], 0).rotate(angle*RadToDeg))

        StartTransf[name] = (pos, angle)

        bone = world.CreateDynamicBody(position=pos,
                                       angle=angle,
                                       allowSleep=False,
                                       fixtures=b2FixtureDef(density=175.0,
                                                             friction=1.0,
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
        return self.bones['torso'].position + Vec2(self.bones['torso'].ms_anchor[1]).rotate(self.bones['torso'].angle * RadToDeg)

    def deactivate(self):
        self.active = False
        for bone in self.bones.values():
            bone.active = False

    def reset(self, mutationRate, mutationScale, model=None):
        self.reward = 0
        self.prevPos = 0
        self.timeAlive = 0
        self.active = True
        self.maxX = 0
        for joint in self.joints:
            joint.motorSpeed = 0
        for name, bone in self.bones.items():
            bone.position = StartTransf[name][0]
            bone.angle = StartTransf[name][1]
            bone.linearVelocity = b2Vec2(0, 0)
            bone.angularVelocity = 0
            bone.active = True

        if model is not None:
            for j, layer in enumerate(self.model.layers):
                self.model.layers[j].set_weights(model.layers[j].get_weights())

        if mutationRate != 0 and mutationScale != 0:
            for j, layer in enumerate(self.model.layers):
                new_weights_for_layer = []
                for weight_array in layer.get_weights():
                    save_shape = weight_array.shape
                    one_dim_weight = weight_array.reshape(-1)

                    for i, weight in enumerate(one_dim_weight):
                        if np.random.random() <= mutationRate:
                            one_dim_weight[i] += mutationScale * np.random.uniform(-0.1, 0.1)
                    new_weights_for_layer.append(one_dim_weight.reshape(save_shape))

                self.model.layers[j].set_weights(new_weights_for_layer)
