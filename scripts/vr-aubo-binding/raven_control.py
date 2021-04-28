#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

from std_msgs.msg import String
import rospy
import math
import geometry_msgs.msg
import tf
import roslib
import tf2_ros
import numpy as np

import os
from absl import app
from absl import flags

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
import pybullet as p

from pyquaternion import Quaternion

#import sensor_msgs.msg
from sensor_msgs.msg import Joy

import threading 


flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')

assets_root = "/home/robot/Downloads/ravens/ravens/environments/assets/"
task_name = "place-red-in-green"
mode = "train"
FLAGS = flags.FLAGS

    
class ViveRobotBridge:
    def __init__(self):
        self.offset = [0,0,0]
        self.offset_flag = 0
        self.grasp = 0
        self.trigger_pressed_event = 0
        self.trigger_released_event = 0
        self.trigger_current_status = 0
        self.trigger_last_status = 0

        self.vive_controller_translation = [0.0, 0.0, 0.0]
        self.vive_controller_rotation = [0.0, 0.0, 0.0, 0.0]
       

        self._joy_sub = rospy.Subscriber('/vive/controller_LHR_FF777F05/joy', 
                                         Joy, self.vive_controller_button_callback,
                                         queue_size=1)
        self.vive_controller_pos_sub = rospy.Subscriber('tarpos_pub', 
                                         geometry_msgs.msg.Transform, self.vive_controller_pos_callback,
                                         queue_size=1)


    def vive_controller_button_callback(self, msg):
        
        if msg.buttons[1] == 1:
            # check for button pressed event
            if self.trigger_last_status == 0:
                self.trigger_pressed_event = 1
            self.trigger_current_status = 1
        else:
            if self.trigger_last_status == 1:
                self.trigger_released_event = 1
            self.trigger_current_status = 0
        # save last status
        self.trigger_last_status = msg.buttons[1]

        if msg.axes[2] == 1.0:
            self.grasp = 1
        else:
            self.grasp = 0
        
    def vive_controller_pos_callback(self, msg):
        
        self.vive_controller_translation[0] = msg.translation.x
        self.vive_controller_translation[1] = msg.translation.y
        self.vive_controller_translation[2] = msg.translation.z

        self.vive_controller_rotation[0] = msg.rotation.x
        self.vive_controller_rotation[1] = msg.rotation.y
        self.vive_controller_rotation[2] = msg.rotation.z
        self.vive_controller_rotation[3] = msg.rotation.w




def main(unused_argv):
    vrb = ViveRobotBridge()
#    vrb.__init__()
    
    rospy.init_node('raven_vive_teleop')


    temp_flag = 0
    
    pre_position = [0,0,0]
    pre_pose = [1, 0, 0, 0]
    temp = [1, 0, 0, 0]

    rate = rospy.Rate(60.0)
    
    limits = 0.02
    
    env = Environment(
      assets_root,
      disp=True,
      hz=60)
    task = tasks.names[task_name]()
    task.mode = mode
    agent = task.oracle(env)
    env.set_task(task)
    obs = env.reset()
    info = None
    ee_pose = ((0.46562498807907104, -0.375, 0.3599780201911926), (0.0, 0.0, 0.0, 1.0))
    ee_position = [0.0, 0.0, 0.0]
    
    while not rospy.is_shutdown():
        """ print(vrb.vive_controller_translation)
        print(vrb.offset_flag)
        print(vrb.grasp)
        continue """
        #p.stepSimulation()
        #p.getCameraImage(480, 320)
        
        if vrb.trigger_pressed_event == 1:
            # get current pose of the end-effector
            ee_position_ref = list(p.getLinkState(env.ur5, env.ee_tip)[4])
            #ee_orientation = p.getLinkState(self.ur5, self.ee_tip)[5]

            vrb.trigger_pressed_event = 0
            print("I--trigger pressed!\n")

        if vrb.trigger_released_event == 1:
            vrb.trigger_released_event = 0
            print("O--trigger released!\n") 

        if vrb.trigger_current_status == 1:
        
            
            
            #ee_orientation = vrb.vive_controller_rotation
            ee_orientation = (0,0,0,1)
            ee_position[0] = ee_position_ref[0] + vrb.vive_controller_translation[1]
            ee_position[1] = ee_position_ref[1] - vrb.vive_controller_translation[0]
            # z axis limit
            z_control = ee_position_ref[2] + vrb.vive_controller_translation[2]
            if z_control < 0.02:
                z_control = 0.02
            ee_position[2] = z_control

            ee_pose = (tuple(ee_position), ee_orientation)
            env.movep(ee_pose)
            marker_head_point = [ee_position[0], ee_position[1], 0.05]
            marker_tail_point = [ee_position[0], ee_position[1], 0.06]
            p.addUserDebugLine( marker_head_point, 
                                marker_tail_point, 
                                lineColorRGB=[1, 0, 0], 
                                lifeTime=0.2, 
                                lineWidth=3)
        
        
        if vrb.grasp == 1:
            env.ee.activate()
        else:
            env.ee.release()
        
        
        p.stepSimulation()
        rate.sleep()
    

        
#    

if __name__ == '__main__':
    app.run(main)