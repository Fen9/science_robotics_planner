#!/usr/bin/env python
import os
import time

import nltk
import numpy as np
import argparse

# ROS
import rospy
import rospkg

# ROS service
from action_manager.srv import *
from robot_data_recorder.srv import *

#ROS message
from robot_data_recorder.msg import *


def begin_record_data():
	rospy.wait_for_service('/begin_record')
	begin_record = rospy.ServiceProxy('/begin_record', beginRecord)
	try:
		resp = begin_record('begin_record')
	except rospy.ServiceException as exc:
		print("Service did not process request"+str(exc))		

def end_record_data():
	rospy.wait_for_service('/end_record')
	end_record = rospy.ServiceProxy('end_record',endRecord)
	try:
		resp = end_record('end record')
		return resp
	except rospy.ServiceException as exc:
		print("Service did not process request"+str(exc))

def main():
	rospy.init_node('record_robot_data')
	with open(rospkg.RosPack().get_path('robot_data_recorder')+'/scripts/sentences.txt', 'r') as f_in:
		data_number = input("Start Number: ")
		pos = 1
		while True:
			sentence = f_in.readline().strip().split(' ')
			print("{}, {}".format(sentence, pos))
			if len(sentence) == 0:
				break
			if int(data_number) > pos+1:
				pos+=1
				continue
			else:
				data_number = input("Data Number: ")

			with open(rospkg.RosPack().get_path('robot_data_recorder')+'/scripts/robot_data/'+'robot_data_'+str(data_number)+'_.csv', 'w') as f_out:
				rospy.wait_for_service('action_manager/exec')
				for action in sentence:
				#record data and execute
					try:
						exec_action = rospy.ServiceProxy('action_manager/exec', execAction)
						# begin_record_data()
						status = exec_action(action)
						# robot_data = end_record_data()
						# for msg in robot_data.data:
						# 	f_out.write('%f,%f,%f,%f,%f,%f,%f,%s\n' % (msg.state.wrench.force.x, msg.state.wrench.force.y, msg.state.wrench.force.z, \
						# 		                                       msg.state.wrench.torque.x, msg.state.wrench.torque.y, msg.state.wrench.torque.z, \
						# 		                                       msg.gap, symbols_dict[action]))

					except rospy.ServiceException, e:
						print("Fail to call action manager")
						exit()
			f_out.close()
			pos+=1	
	f_in.close()		


symbols_dict = {	
	'approach':'0', 	#0
	'grasp':'1',		#1	
	'ungrasp':'2',		#2
	'twist':'3',		#3
	'reverse':'4',		#4
	'push':'5',			#5
	'unpush':'6',		#6
	'pinch':'7',		#7
	'unpinch':'8',		#8
	'pull':'9',			#9
	}

if __name__ == '__main__':
	main()
