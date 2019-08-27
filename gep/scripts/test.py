#!/usr/bin/env python
import os
import time

import nltk
import numpy as np

# ROS
import rospy
import rospkg

# Generalized Earley Parser
import grammarutils
import generalizedearley

from transition_srv.srv import *
from gep.srv import *
from action_manager.srv import *


def execute_robot_action(next_action):
	rospy.wait_for_service('action_manager/exec')
	try:
		exec_action = rospy.ServiceProxy('action_manager/exec', execAction)
		status = exec_action(next_action)
		return status, next_action
	except rospy.ServiceException, e:
		print("Fail to call action manager")

def get_new_frame():
	rospy.wait_for_service('get_transition')
	transition_prob = rospy.ServiceProxy('get_transition', transition)

	try:
		resp = transition_prob(True, False)
	except rospy.ServiceException, e:
		print("Fail to call transition srv")

	new_frame_prob = [float(prob) for prob in list(resp.next)]
	return new_frame_prob

def test():
	rospy.wait_for_service('planner_server/plan')
	planning = rospy.ServiceProxy('planner_server/plan', planAction)

	new_frame_prob = get_new_frame()
	# print("next most possible action:")
	# idx = input()
	# new_frame_prob = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
	# new_frame_prob = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
	# new_frame_prob[int(idx)] = 0.88

	try:
		resp = planning(new_frame_prob, False)
		print(resp.action_seq)
	except rospy.ServiceException as exc:
		print("Service did not process request: " + str(exc))

symbols = [	
	'end',
	'approach',
	'move',
	'grasp_left',
	'grasp_right',
	'ungrasp_left',
	'ungrasp_right',
	'twist',
	'push',
	'neutral',
	'pull',
	'pinch',
	'unpinch'
	]

if __name__ == "__main__":
	# new_frame_prob = np.zeros(13)
	# print(new_frame_prob.shape)
	# for i in range(0, new_frame_prob.shape[0]):
	# 	new_frame_prob[i] = 1e-20
	# print(new_frame_prob)
	test()
