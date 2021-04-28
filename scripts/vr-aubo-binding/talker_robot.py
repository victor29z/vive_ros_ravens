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

FLAGS = flags.FLAGS
    
class ViveRobotBridge:
    def __init__(self):
        self.offset = [0,0,0]
        self.offset_flag = 0
        self.grasp = 0
        self.pub = rospy.Publisher('tarpos_pub', geometry_msgs.msg.Transform,
                                   queue_size=1)
#        print("subscribe controller button events")
        self._joy_sub = rospy.Subscriber('/vive/controller_LHR_FF777F05/joy', 
                                         Joy, self.vive_controller_button_callback,
                                         queue_size=1)
#        self.tf_thread = threading.Thread(target = self.TransformListener, args=())
#        print("starting thread")
#        self.tf_thread.start()

    def vive_controller_button_callback(self, msg):
        if msg.buttons[1] == 1:
#            print("controller events")
            self.offset_flag = 1
        else:
            self.offset_flag = 0

        if msg.axes[2] == 1.0:
            self.grasp = 1
        else:
            self.grasp = 0
        
    def publish(self, msg):
        self.pub.publish(msg)



def main(unused_argv):
    vrb = ViveRobotBridge()
#    vrb.__init__()
    
    rospy.init_node('vive_listener')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
#    pub = rospy.Publisher('tarpos_pub', geometry_msgs.msg.Transform, queue_size=1)
#    _joy_sub = rospy.Subscriber('/vive/controller_LHR_FFFF3F47/joy', Joy, vive_controller_button_callback, queue_size=1)
    
    temp_flag = 0
    
    pre_position = [0,0,0]
    pre_pose = [1, 0, 0, 0]
    temp = [1, 0, 0, 0]

    rate = rospy.Rate(100.0)
    
    limits = 0.02
    
    env = Environment(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
    task = tasks.names[FLAGS.task]()
    task.mode = FLAGS.mode
    agent = task.oracle(env)
    env.set_task(task)
    obs = env.reset()
    info = None
    ee_pose = ((0.46562498807907104, -0.375, 0.3599780201911926), (0.0, 0.0, 0.0, 1.0))
    while not rospy.is_shutdown():
        #act = agent.act(obs, info)
        
        #obs, reward, done, info = env.step(act)
        

        env.movep(ee_pose)
        if vrb.grasp == 1:
            env.ee.activate()
        else:
            env.ee.release()
        
        try:
            trans = tfBuffer.lookup_transform('world', 'controller_LHR_FF777F05', rospy.Time())
#            rospy.loginfo(trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("error")
            rate.sleep()
            continue
        

        if vrb.offset_flag == 1:
            if temp_flag == 0:
#                vrb.offset[0] = trans.transform.translation.x
#                vrb.offset[1] = trans.transform.translation.y
#                vrb.offset[2] = trans.transform.translation.z
#                print(vrb.offset)
                pre_position[0]=trans.transform.translation.x
                pre_position[1]=trans.transform.translation.y
                pre_position[2]=trans.transform.translation.z
                
                pre_pose[0] = trans.transform.rotation.x
                pre_pose[1] = trans.transform.rotation.y
                pre_pose[2] = trans.transform.rotation.z
                pre_pose[3] = trans.transform.rotation.w
            else:
                msg = geometry_msgs.msg.Transform()            
#                msg.translation.x = (trans.transform.translation.x-pre_position[0])/10
#                msg.translation.y = (trans.transform.translation.y-pre_position[1])/10
#                msg.translation.z = (trans.transform.translation.z-pre_position[2])/10
                
#                compute delta distance
                msg.translation.x = trans.transform.translation.x-pre_position[0]
                msg.translation.y = trans.transform.translation.y-pre_position[1]
                msg.translation.z = trans.transform.translation.z-pre_position[2]
                
#                if msg.translation.x >0.02:
#                    msg.translation.x = 0.02                
#                if msg.translation.y >0.02:
#                    msg.translation.y = 0.02                
#                if msg.translation.z >0.02:
#                    msg.translation.z = 0.02                     
#                if msg.translation.x <-0.02:
#                    msg.translation.x = -0.02                
#                if msg.translation.y <-0.02:
#                    msg.translation.y = -0.02                
#                if msg.translation.z <-0.02:
#                    msg.translation.z = -0.02
                                        
                if msg.translation.x >limits:
                    msg.translation.x = limits                
                if msg.translation.y >limits:
                    msg.translation.y = limits                
                if msg.translation.z >limits:
                    msg.translation.z = limits                     
                if msg.translation.x <-limits:
                    msg.translation.x = -limits                
                if msg.translation.y <-limits:
                    msg.translation.y = -limits                
                if msg.translation.z <-limits:
                    msg.translation.z = -limits
                
                print(msg.translation)
                
#                temp[0] = trans.transform.rotation.x
#                temp[1] = trans.transform.rotation.y
#                temp[2] = trans.transform.rotation.z
#                temp[3] = trans.transform.rotation.w
#                
#                
#                
#                q = Quaternion(pre_pose) * Quaternion(temp).inverse
#                
#                
#                msg.rotation.x = q.x
#                msg.rotation.y = q.y
#                msg.rotation.z = q.z
#                msg.rotation.w = q.w#                
                
                msg.rotation.x = trans.transform.rotation.x
                msg.rotation.y = trans.transform.rotation.y
                msg.rotation.z = trans.transform.rotation.z
                msg.rotation.w = trans.transform.rotation.w
                
                ee_position = list(p.getLinkState(env.ur5, env.ee_tip)[4])
                #ee_orientation = p.getLinkState(self.ur5, self.ee_tip)[5]
                ee_orientation = (0,0,0,1)
                ee_position[0] = ee_position[0] + msg.translation.y
                ee_position[1] = ee_position[1] - msg.translation.x
                # z axis limit
                z_control = ee_position[2] + msg.translation.z
                if z_control < 0.02:
                    z_control = 0.02
                ee_position[2] = z_control

                ee_pose = (tuple(ee_position), ee_orientation)
                
                # rectified quaternion 
                
#                theta_x = np.arcsin(2*(trans.transform.rotation.w*trans.transform.rotation.y 
#                                       - trans.transform.rotation.z*trans.transform.rotation.x))
#                temp = (trans.transform.rotation.x**2 + trans.transform.rotation.z**2)**0.5
#                z_v = trans.transform.rotation.z / temp
#                x_v = trans.transform.rotation.x / temp
#                msg.rotation.x = x_v * np.sin(theta_x)#0#np.sin(0.5*theta_x)
#                msg.rotation.y = 0#y_v * np.sin(theta_x)#0
#                msg.rotation.z = z_v * np.sin(theta_x)
#                msg.rotation.w = trans.transform.rotation.w
#                
#                msg.translation.x = 0
#                msg.translation.y = 0
#                msg.translation.z = 0
#                
#                msg.rotation.x = 0
#                msg.rotation.y = 0
#                msg.rotation.z = 0
#                msg.rotation.w = 1
                
                print(msg.rotation)
                
                vrb.pub.publish(msg)
                
                pre_position[0]=trans.transform.translation.x
                pre_position[1]=trans.transform.translation.y
                pre_position[2]=trans.transform.translation.z
            

        temp_flag = vrb.offset_flag
        rate.sleep()
        
#        if vrb.offset_flag == 1:
##        print(vrb.offset_flag)
#            msg = geometry_msgs.msg.Transform()            
#            msg.translation.x = trans.transform.translation.x-vrb.offset[0]
#            msg.translation.y = trans.transform.translation.y-vrb.offset[1]
#            msg.translation.z = trans.transform.translation.z-vrb.offset[2]+0.8
#            msg.rotation.x = trans.transform.rotation.x
#            msg.rotation.y = trans.transform.rotation.y
#            msg.rotation.z = trans.transform.rotation.z
#            msg.rotation.w = trans.transform.rotation.w
        
#            vrb.pub.publish(msg)

        
#    

if __name__ == '__main__':
    app.run(main)