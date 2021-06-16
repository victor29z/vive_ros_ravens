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
import time
from absl import app
from absl import flags

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
import pybullet as p

from pyquaternion import Quaternion

#import sensor_msgs.msg
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JointState

import threading 
import PyKDL as kdl

from matplotlib import pyplot as plt

import train_bc
import torch
from torch.utils import data
from torch.autograd import Variable

import torch.nn as nn



flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')

assets_root = "/home/robot/Downloads/ravens/ravens/environments/assets/"
dataset_root = "/data/ravens_bc_rollout/"
model_save_rootdir = "/data/trained_model/"
#task_name = "place-red-in-green"
task_name = "block-insertion-nofixture"
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

        if msg.buttons[2] == 1:
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

class RobotControlInterface:
    def __init__(self, env):
        self.joint_states = [0,0,0,0,0,0]
        self.ik_result_sub = rospy.Subscriber(
            '/joint_states', 
            JointState, 
            self.ik_result_callback,
            queue_size=1)
        self.ee_tarpos_pub = rospy.Publisher(
            'ee_tarpos', 
            geometry_msgs.msg.Transform,
            queue_size=1)

        self.joint_state_pub = rospy.Publisher(
            'joint_states', 
            JointState,
            queue_size=1)

        self.env = env

    def ik_result_callback(self, msg):
        """joint states of external ik solver"""
        self.joint_states = msg.position
        self.movej_fast(self.joint_states)
        
    def movej_fast(self, targj):
        
        gains = np.ones(len(self.env.joints))
        p.setJointMotorControlArray(
            bodyIndex=self.env.ur5,
            jointIndices=self.env.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=targj,
            positionGains=gains)
        p.stepSimulation()

    def publish_pose_message(self, pose):
        msg = geometry_msgs.msg.Transform()    

        position = pose[0]
        orientation = pose[1]
        msg.translation.x = position[0]
        msg.translation.y = position[1]
        msg.translation.z = position[2]

        msg.rotation.x = orientation[0]
        msg.rotation.y = orientation[1]
        msg.rotation.z = orientation[2]
        msg.rotation.w = orientation[3]
        print(msg)

        self.ee_tarpos_pub.publish(msg)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.env.ur5,
            endEffectorLinkIndex=self.env.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.env.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        self.joint_states = joints
        return joints
    
    def publish_joint_states_msg(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time().now()
        joint_state_msg.name = ['shoulder_pan_joint',
                                'shoulder_lift_joint',
                                'elbow_joint',
                                'wrist_1_joint',
                                'wrist_2_joint',
                                'wrist_3_joint']
        joint_state_msg.position =  self.joint_states      
        self.joint_state_pub.publish(joint_state_msg)         

class teleop_agent:
    def __init__(self, env) -> None:
        self.vrb = ViveRobotBridge()
        self.pose = ((0.0, 0.0, 0.0),(0.0, 0.0, 0.0, 1.0))
        self.grasp = 0
        
        self.env = env
    def act(self):
        action = {}
        ee_pose = ((0.46562498807907104, -0.375, 0.3599780201911926), (0.0, 0.0, 0.0, 1.0))
        ee_position = [0.0, 0.0, 0.0]
        #update grasp state
        if self.vrb.grasp == 1:
            action['grasp'] = 1
        else:
            action['grasp'] = 0
        
        if self.vrb.trigger_pressed_event == 1:
            # get current pose of the end-effector
            self.ee_position_ref = list(p.getLinkState(self.env.ur5, self.env.ee_tip)[4])
            #ee_orientation = p.getLinkState(self.ur5, self.ee_tip)[5]

            self.vrb.trigger_pressed_event = 0
            print("I--trigger pressed!\n")

        if self.vrb.trigger_released_event == 1:
            self.vrb.trigger_released_event = 0
            print("O--trigger released!\n") 

        if self.vrb.trigger_current_status == 1:    
            vive_rotation = self.vrb.vive_controller_rotation
            ee_rotation = kdl.Rotation.Quaternion(vive_rotation[0],vive_rotation[1],vive_rotation[2],vive_rotation[3])
            r1 = kdl.Rotation.RotZ(-math.pi/2)
            ee_rotation = r1 * ee_rotation
            ee_rotation.DoRotX(math.pi/2)
            
            ee_rotation.DoRotZ(math.pi/2)

            ee_rotation.DoRotY(math.pi)
            
            ee_orientation = ee_rotation.GetQuaternion()
            
   
            #ee_orientation = (0,0,0,1)
            ee_position[0] = self.ee_position_ref[0] + self.vrb.vive_controller_translation[1]
            ee_position[1] = self.ee_position_ref[1] - self.vrb.vive_controller_translation[0]
            # z axis limit
            z_control = self.ee_position_ref[2] + self.vrb.vive_controller_translation[2]
            if z_control < 0.02:
                z_control = 0.02
            ee_position[2] = z_control

            self.pose = (tuple(ee_position), tuple(ee_orientation))
            action['pose'] = (tuple(ee_position), tuple(ee_orientation))
        
        
            return action
        else:
            return None



        
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=64,
                            kernel_size=7,
                            stride=2
                            ),
            #torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
            #torch.nn.MaxPool2d(kernel_size = 2)
        ).cuda()
        #self.conv1 = self.conv1.cuda()
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=32,
                            kernel_size=1
                            ),

            torch.nn.ReLU()
        ).cuda()
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3
                            ),

            torch.nn.ReLU()
        ).cuda()
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3
                            ),

            torch.nn.ReLU()
        ).cuda()
        self.mlp1 = torch.nn.Linear(73*53*32,100).cuda()
        self.mlp2 = torch.nn.Linear(100,50).cuda()
        self.mlp3 = torch.nn.Linear(50,8).cuda()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.contiguous().view(x.size(0),-1))
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x

        


def main(unused_argv):
    
    
    
    
    rospy.init_node('raven_vive_teleop')
     

    temp_flag = 0
    
    pre_position = [0,0,0]
    pre_pose = [1, 0, 0, 0]
    temp = [1, 0, 0, 0]
    # same loop rate as gym environment
    rate = rospy.Rate(60.0)
    
    
    env = Environment(
      assets_root,
      disp=True,
      hz=60)
    
    task = tasks.names[task_name]()
    task.mode = mode

    
    env.set_task(task)
    obs = env.reset()
    info = None
    #agent = teleop_agent(env)
    dataset = Dataset(os.path.join(dataset_root, f'ravens-bc-rollout-{time.time_ns()}'))
    seed = 0

    br = tf.TransformBroadcaster()
    rci = RobotControlInterface(env)

    episode = []
    reward = 0
    done = False

    plt.ion()
    im_color = obs['color'][0]
    im_depth = obs['depth'][0]
    img_draw = plt.imshow(im_depth, interpolation='nearest')

    agent = train_bc.BC_Agent()
    agent.load_pretrained_model(model_save_rootdir + "model_500_epoch")
    agent.act(obs)
    f = open(model_save_rootdir+ "rollout_log.txt","w+")
    episode_steps = 0
    n_episode = 1
    while not rospy.is_shutdown():
  
        #p.stepSimulation()
        #p.getCameraImage(480, 320)
        keys = p.getKeyboardEvents()
        if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_RELEASED:
            print("reset env")
            episode = []

            obs = env.reset()
        
        action = agent.act(obs)
        #if action != None:
        #    print(action)
    
        
        obs, reward, done, info = env.step_simple(action)
        # im_depth = obs['depth'][0]
        # img_draw.set_data(im_depth)
        # plt.pause(0.00000001)
        # plt.show()
        # print( im_color.shape)

        # print(obs['color'])
        s = "Episode:%d, steps:%d"%(n_episode, episode_steps)
        print(s)
        if(done):
            n_episode += 1
            episode_steps = 0
            s += ", succeed!\n" 
            f.write(s)
            f.flush()
            seed += 1
            dataset.add(seed, episode)
            episode = []

        else:
            episode_steps += 1

        if episode_steps > 100:
            episode = []
            episode_steps = 0
            n_episode += 1

            obs = env.reset()
            s += ", failed!\n" 
            f.write(s)
            f.flush()
        


        

        
        rate.sleep()

    

        
#    

if __name__ == '__main__':
    app.run(main)