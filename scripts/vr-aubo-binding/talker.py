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

from pyquaternion import Quaternion

#import sensor_msgs.msg
from sensor_msgs.msg import Joy

import threading 

    
    
class ViveRobotBridge:
    def __init__(self):
        self.offset = [0,0,0]
        self.offset_flag = 0
        self.grasp = 0

        self.trigger_pressed_event = 0
        self.trigger_released_event = 0
        self.trigger_current_status = 0
        self.trigger_last_status = 0
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
        
    def publish(self, msg):
        self.pub.publish(msg)




if __name__ == '__main__':

    vrb = ViveRobotBridge()
#    vrb.__init__()
    
    rospy.init_node('vive_publisher')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
#    pub = rospy.Publisher('tarpos_pub', geometry_msgs.msg.Transform, queue_size=1)
#    _joy_sub = rospy.Subscriber('/vive/controller_LHR_FFFF3F47/joy', Joy, vive_controller_button_callback, queue_size=1)
    
    temp_flag = 0
    
    pre_position = [0,0,0]
    pre_pose = [1, 0, 0, 0]
    temp = [1, 0, 0, 0]

    rate = rospy.Rate(50.0)
    
    limits = 0.015
    
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('world', 'controller_LHR_FF777F05', rospy.Time())
#            rospy.loginfo(trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("error")
            rate.sleep()
            continue
        if vrb.trigger_pressed_event == 1:
            pre_position[0]=trans.transform.translation.x
            pre_position[1]=trans.transform.translation.y
            pre_position[2]=trans.transform.translation.z
            
            pre_pose[0] = trans.transform.rotation.x
            pre_pose[1] = trans.transform.rotation.y
            pre_pose[2] = trans.transform.rotation.z
            pre_pose[3] = trans.transform.rotation.w

            vrb.trigger_pressed_event = 0
            print("I--trigger pressed!\n")
        if vrb.trigger_released_event == 1:
            vrb.trigger_released_event = 0
            print("O--trigger released!\n")
        if vrb.trigger_current_status == 0:
            vrb.trigger_pressed_event = 0
            vrb.trigger_released_event = 0
        else:
            msg = geometry_msgs.msg.Transform()            

    #       compute delta distance
            
            msg.translation.x = trans.transform.translation.x-pre_position[0]
            msg.translation.y = trans.transform.translation.y-pre_position[1]
            msg.translation.z = trans.transform.translation.z-pre_position[2]
            
            msg.rotation.x = trans.transform.rotation.x
            msg.rotation.y = trans.transform.rotation.y
            msg.rotation.z = trans.transform.rotation.z
            msg.rotation.w = trans.transform.rotation.w
            
            print(msg.translation)
            #print(msg.rotation)
            
            vrb.pub.publish(msg)
        rate.sleep()

#    