from pygame import Vector2 as Vec2
import numpy as np
from Box2D import *


RadToDeg = 180/b2_pi


def CreateBone(world, parent, size=(1, 0.2), angle=0, pos=(0, 6), anchor0=-0.95, anchor1=0.95, parentAnchor=1, thisAnchor=0, angleLow=0, angleHigh=0, maxTorque=10):
    angle = angle if parent is None else parent.angle
    pos = pos if parent is None else (parent.position +
                                      Vec2(parent.ms_anchor[parentAnchor][0] + size[0], 0).rotate(angle*RadToDeg))

    bone = world.CreateDynamicBody(position=pos,
                                   angle=angle,
                                   allowSleep=False,
                                   fixtures=b2FixtureDef(density=1.0,
                                                         friction=0.75,
                                                         shape=b2PolygonShape(box=size),
                                                         categoryBits=0x0004, maskBits=0x0002))
    bone.ms_anchor = [Vec2(anchor0 * size[0], 0),
                      Vec2(anchor1 * size[0], 0)]
    if parent is None:
        return bone

    parent.ms_joints.append(world.CreateRevoluteJoint(bodyA=parent,
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
    return bone


class Actor:
    def __init__(self, world):
        self.bones = {}
        self.bones['torso'] = world.CreateBone(None, size=(0.6, 0.2), angle=-b2_pi*0.5)

        self.bones['thigh1'] = world.CreateBone(self.bones['torso'], angleLow=-b2_pi*0.25, angleHigh=b2_pi*0.5, size=(0.4, 0.15))
        self.bones['thigh2'] = world.CreateBone(self.bones['torso'], angleLow=-b2_pi*0.25, angleHigh=b2_pi*0.5, size=(0.4, 0.15))
        self.bones['crus1'] = world.CreateBone(self.bones['thigh1'], angleLow=-b2_pi*0.9, angleHigh=0, size=(0.35, 0.1))
        self.bones['crus2'] = world.CreateBone(self.bones['thigh2'], angleLow=-b2_pi*0.9, angleHigh=0, size=(0.35, 0.1))
        self.bones['foot1'] = world.CreateBone(self.bones['crus1'], angleLow=0, angleHigh=b2_pi*0.65, size=(0.2, 0.05))
        self.bones['foot2'] = world.CreateBone(self.bones['crus2'], angleLow=0, angleHigh=b2_pi*0.65, size=(0.2, 0.05))

    def getInputArray(self):
        inputs = []
        rootPos = self.bones['torso'].position + Vec2(self.bones['torso'].ms_anchor[1]).rotate(self.bones['torso'].angle * RadToDeg)
        inputs.append(rootPos.y)
        for bone in list(self.bones.values())[1:]:  # pomijamy roota
            boneRelPos = (bone.position + Vec2(bone.ms_anchor[1]).rotate(bone.angle * RadToDeg)) - rootPos
            boneVel = bone.linearVelocity + Vec2(0, bone.ms_anchor[1][0] * bone.angularVelocity).rotate(bone.angle * RadToDeg)
            inputs += boneRelPos
            inputs += boneVel
        for e in inputs:
            print(e)
        print("")

        return np.array(inputs)

    @staticmethod
    def CreateActor(world):
        return Actor(world)
