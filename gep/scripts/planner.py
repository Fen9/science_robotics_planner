#!/usr/bin/env python
import os
import time

import nltk
import numpy as np
import argparse

# ROS
import rospy
import rospkg

# Generalized Earley Parser
import grammarutils
import generalizedearley

# ROS service
from transition_srv.srv import *
from joint_transition_srv.srv import *
from action_manager.srv import *

#ROS message
from gep.srv import *
from gep.msg import *

# parser = argparse.ArgumentParser()
# parser.add_argument('parser', type=str, default="gep")
# parser.add_argument('rule_path', type=str, default='grammar/grammar_scirot_v2.txt')
# args = parser.parse_args()


def read_infuced_grammar(path, symbols):
	with open(path) as f:
		rules = [rule.strip() for rule in f.readlines()]
		rules.insert(0, 'GAMMA -> S [1.0]')
		symbols_index = dict()
		for s in symbols:
			symbols_index[s] = symbols.index(s)
		grammar_rules = grammarutils.get_pcfg(rules, index=True, mapping=symbols_index)
		grammar = nltk.PCFG.fromstring(grammar_rules)
		return grammar


class Planner:
	def __init__(self, rule_path, symbols):
		self._grammar = read_infuced_grammar(rule_path, symbols)
		self._symbols = symbols
		self._bottom_up_prob_matrix = []
		self._generalized_earley_parser = generalizedearley.GeneralizedEarley(self._grammar)
		self._current_action_seq = []
		self._exec_status = False

	def add_new_frame(self, new_frame):
		self._bottom_up_prob_matrix.append(new_frame)
		return self._bottom_up_prob_matrix

	def update_prob_matrix(self):
		new_frame_prob = np.zeros(10)
		# print(new_frame_prob.shape)
		for i in range(0, new_frame_prob.shape[0]):
			if i == self._current_action_seq[-1]:
				new_frame_prob[i] = 1.0
			else:
				new_frame_prob[i] = 1e-20
		new_frame_prob = new_frame_prob / np.linalg.norm(new_frame_prob)
		# new_frame_prob = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
		# new_frame_prob[self._current_action_seq[-1]] = 0.88
		self._bottom_up_prob_matrix[-1] = new_frame_prob.tolist()

	def set_exec_status(self, status):
		self._exec_status = status

	def parse(self, new_frame):
		prob_matrix = self.add_new_frame(new_frame)
		prob_matrix = np.asarray(prob_matrix, dtype=np.float32)
		best_parse, prob = self._generalized_earley_parser.parse(prob_matrix)
		return best_parse, prob

	def plan_next_action(self, next_frame_prob):
		print("plan next action")
		best_parse, prob = planner.parse(next_frame_prob)
		print("best parse sentense: {}".format(best_parse))
		planner._current_action_seq = [int(action) for action in list(best_parse.split(' '))]
		planner.update_prob_matrix()
		return planner._current_action_seq[-1]

	def record_robot_sensing(self):
		rospy.wait_for_service('/begin_record')
		begin_record = rospy.ServiceProxy('begin_record', beginRecord)
		try:
			resp = begin_record('begin record')
		except rospy.ServiceException as exc:
			print("Service did not process request"+str(exc))

	def get_sensing_resp(self):
		rospy.wait_for_service('/end_record')
		end_record = rospy.ServiceProxy('end_record',endRecord)
		try:
			resp = end_record('end record')
			return resp
		except rospy.ServiceException as exc:
			print("Service did not process request"+str(exc))


	def process_sensing_resp(self, resp, win_size=10):
		msg_lst = []
		for msg in resp.data[-win_size:]:
			# msg_lst.append([msg.state.pose.position.x, msg.state.pose.position.y, msg.state.pose.position.z,
			# 	msg.state.pose.orientation.x, msg.state.pose.orientation.y, msg.state.pose.orientation.z, msg.state.pose.orientation.w,
			# 	msg.state.twist.linear.x, msg.state.twist.linear.y, msg.state.twist.linear.z,
			# 	msg.state.twist.angular.x, msg.state.twist.angular.y, msg.state.twist.angular.z,
			# 	msg.state.wrench.force.x, msg.state.wrench.force.y, msg.state.wrench.force.z,
			# 	msg.state.wrench.torque.x, msg.state.wrench.torque.y, msg.state.wrench.torque.z,
			# 	msg.gap])
			print('msg recv')
			print([msg.state.wrench.force.x, msg.state.wrench.force.y, msg.state.wrench.force.z,
				msg.gap*255.0])
			msg_lst.append([msg.state.wrench.force.x, msg.state.wrench.force.y, msg.state.wrench.force.z,
				msg.gap*255.0])
		print('others')
		return np.mean(np.asarray(msg_lst), axis=0).tolist()

	def execute_robot_action(self, next_action):
		rospy.wait_for_service('action_manager/exec')
		try:
			exec_action = rospy.ServiceProxy('action_manager/exec', execAction)
			#start record
			self.record_robot_sensing()
			#start execute robot actions
			status = exec_action(next_action)
			#end record
			resp = self.get_sensing_resp()
			return status, self.process_sensing_resp(resp)
		except rospy.ServiceException, e:
			print("Fail to call action manager")
			exit()

	def reset(self):
		self._bottom_up_prob_matrix = []


def gep_plan_execute():
	rospy.init_node('gep_plan_execute')
	print('Start GEP planning and executing robot actions')
	termination_action = "pull"
	max_termination_step = 12
	next_frame = [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
	current_step = 0
	current_action = ' '
	while not rospy.is_shutdown():
		next_action = planner.plan_next_action(next_frame)
		if next_action == current_action:
			rospy.wait_for_service('transition_srv/pred_next_action')
			pred_next_action = rospy.ServiceProxy('transition_srv/pred_next_action', predAction)
			next_frame = np.array(pred_next_action(robot_sensing_feedback, next_action).next_action_prob).tolist()
			print('next action prob:{}'.format(next_frame))
			continue
		print("next action: {}, current_parsed_tree: {}".format(symbols[next_action], planner._current_action_seq))
		if current_step < max_termination_step:
			exec_status, robot_sensing_feedback = planner.execute_robot_action(symbols[next_action])
			if not exec_status:
				break
			rospy.wait_for_service('transition_srv/pred_next_action')
			pred_next_action = rospy.ServiceProxy('transition_srv/pred_next_action', predAction)
			next_frame = np.array(pred_next_action(robot_sensing_feedback, next_action).next_action_prob).tolist()
			print('next action prob:{}'.format(next_frame))
		else:
			break
		current_step += 1
		current_action = next_action
		if symbols[current_action] == termination_action:
			print('end')
			break
	return

def sensory_plan_execute():
	rospy.init_node('sensing_plan_execute')
	print('Start Sensory planning and executing robot actions')
	termination_action = 'pull'
	max_termination_step = 20
	current_step = 0
	next_action = 0
	current_action = next_action
	while not rospy.is_shutdown():
		if current_step < max_termination_step:
			exec_status, robot_sensing_feedback = planner.execute_robot_action(symbols[next_action])
			if not exec_status:
				break
			rospy.wait_for_service('transition_srv/pred_next_action')
			pred_next_action = rospy.ServiceProxy('transition_srv/pred_next_action', predAction)
			next_frame = np.array(pred_next_action(robot_sensing_feedback, next_action).next_action_prob)
			next_action = np.argwhere(np.random.uniform(0, 1) < np.cumsum(next_frame)).squeeze(-1)[0]
			print(next_action)
		else:
			break
		print('next action prob:{}'.format(next_frame))
		if next_action == current_action:
			rospy.wait_for_service('transition_srv/pred_next_action')
			pred_next_action = rospy.ServiceProxy('transition_srv/pred_next_action', predAction)
			next_frame = np.array(pred_next_action(robot_sensing_feedback, next_action).next_action_prob).tolist()
		current_step += 1
		current_action = next_action
	return

def symbolic_plan_execute():
	rospy.init_node('symbolic_plan_execute')
	print('Start Symbolic planning and executing robot actions')
	termination_action = '9'
	max_trails = 37
	with open('sampled_sentence.txt', 'w') as fid:
		for current_trail in range(0, 37):
			current_seq = []
			current_seq.append('0')
			while True:
				# grammarutils.earley_predict(planner._grammar, current_seq)
				next_actions, next_action_probs = grammarutils.earley_predict(planner._grammar, current_seq)
				print('next actions')
				print(next_actions)
				print('probs')
				print(next_action_probs)

				next_action_probs = np.array(next_action_probs)/sum(next_action_probs)
				next_action_idx = np.argwhere(np.random.uniform(0, 1) < np.cumsum(np.array(next_action_probs))).squeeze(-1)
				if len(next_action_idx) > 0:
					current_seq.append(str(next_actions[next_action_idx[0]]))
				else:
					for symbol_idx in current_seq:
						symbol = symbols[int(symbol_idx)]
						fid.write(symbol+',')
					fid.write('\n')
					print('can not parse anymore')
					break
	fid.close()


def init_planner(symbols):
	planner = Planner(rospkg.RosPack().get_path('gep')+ '/grammar/grammar_scirobot_v3.txt', symbols)
	return planner

def test():
	planner._bottom_up_prob_matrix.append([0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
	planner._bottom_up_prob_matrix.append([0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
	planner._bottom_up_prob_matrix.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01])
	planner._bottom_up_prob_matrix.append([0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
	planner._bottom_up_prob_matrix.append([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01])
	planner._bottom_up_prob_matrix.append([0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
	planner.plan_next_action([0.000000001, 3.1707122616353445e-06, 0.007722804322838783, 0.00018654702580533922, 0.6335882544517517, 0.01591411791741848, 0.33866605162620544, 2.503144003185298e-07, 8.232692039200629e-08, 0.003918794449418783])

symbols_dict = {	
	'approach': '0', 	#0
	'grasp': '1',		#1	
	'ungrasp': '2',		#2
	'twist': '3',		#3
	'reverse': '4',		#4
	'push': '5',		#5
	'unpush': '6',		#6
	'pinch': '7',		#7
	'unpinch': '8',		#8
	'pull': '9',		#9
	}

symbols = [	
	'approach', 	#0
	'grasp',		#1	
	'ungrasp',		#2
	'twist',		#3
	'reverse',		#4
	'push',			#5
	'unpush',		#6
	'pinch',		#7
	'unpinch',		#8
	'pull',			#9
	]
planner = init_planner(symbols)
if __name__ == "__main__":
	# gep_plan_execute()
	# sensory_plan_execute()
	symbolic_plan_execute()
	#test()