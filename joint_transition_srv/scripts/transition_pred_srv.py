#!/usr/bin/env python
# Sys
import os
import time
import numpy as np
import argparse

#ROS
import rospy
import rospkg

#Pytorch
import torch
import torch.nn.functional as F

#Model pkg
import transition_model

#ROS msg & srv
from joint_transition_srv.srv import *
from std_msgs.msg import String
from baxter_core_msgs.msg import EndpointState


parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--model', type=str, default='WReN')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_path', type=str, default='/results/checkpoint/')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--win_size', type=int, default=10)

args = parser.parse_args()
args.in_dim = 80
args.robot_dim = 4
args.embed_dim = 8
args.action_dim = 10
args.cuda = False

class transition_pred:
    def __init__(self, args):
        self.model = transition_model.joint_transition_model(args)
        self.model_path = args.model_path
        self.load_model()

    def load_model(self):
        state_dict = torch.load(rospkg.RosPack().get_path('joint_transition_srv')+'/scripts'+self.model_path+'model_usable_1.pth')
        # state_dict = torch.load(rospkg.RosPack().get_path('joint_transition_srv')+'/scripts'+self.model_path+'model_usable_2.pth')
        # state_dict = torch.load(rospkg.RosPack().get_path('joint_transition_srv')+'/scripts'+self.model_path+'model.pth')
        self.model.human_encoder.load_state_dict(state_dict['human_encoder'])
        self.model.human_decoder.load_state_dict(state_dict['human_decoder'])
        self.model.action_predictor.load_state_dict(state_dict['action_predictor'])
        self.model.robot_mapping.load_state_dict(state_dict['robot_mapping'])
        self.model.recon_optimizer.load_state_dict(state_dict['recon_optim'])
        self.model.pred_optimizer.load_state_dict(state_dict['pred_optim'])

        # self.model.state_dict = torch.load(rospkg.RosPack().get_path('joint_transition_srv')+'/scripts'+self.model_path+'model.pth')
        # self.model.state_dict = torch.load(self.model_path+'model.pth')

    def pred_next_action(self, robot_state, current_action):
        robot_state = np.array([robot_state])
        robot_state = torch.from_numpy(robot_state).type(torch.float)
        vec = np.zeros((10, ))
        vec[current_action] = 1
        current_action_vec = torch.from_numpy(vec).type(torch.float).view(-1, 10)
        next_action_prob = F.softmax(self.model(None, robot_state, current_action_vec, eval=True), dim=-1)
        return next_action_prob.cpu().detach().numpy().tolist()


pred_model = transition_pred(args)

def compute_next_action_prob(req):
    robot_state = req.robot_state
    current_action = req.current_action
    next_action_prob = pred_model.pred_next_action(robot_state, current_action)
    return next_action_prob

def transition_server():
    rospy.init_node('transition')
    transition_service = rospy.Service('transition_srv/pred_next_action', predAction, compute_next_action_prob)
    print("transition_srv is ready")
    rospy.spin()

def test():
    data = [[-7.025805,	0.552432,	8.271596,   0.521569*255],
            [-7.308488,	0.737033,	8.378974,	0.521569*255],
            [-6.851611,	0.52472,	8.154445,	0.521569*255],
            [-7.015872,	0.495913,	8.28823,	0.521569*255],
            [-6.846816,	0.427428,	8.337986,	0.521569*255],
            [-6.85888,	0.486167,	8.263615,	0.521569*255],
            [-7.055781,	0.575782,	8.222486,	0.521569*255],
            [-6.814876,	0.444718,	8.270287,	0.521569*255],
            [-6.350617,	0.215631,	8.430635,	0.521569*255]]





    data = torch.from_numpy(np.array(data)).type(torch.float)
    data = torch.mean(data, dim=0, keepdim=True)

    # data = [[-1.6529,  3.2924, 10.1368,  0.1758, -1.7864,  0.2687,  0.3765]]
    # data = torch.from_numpy(np.array(data)).type(torch.float)

    current_action = 1
    vec = np.zeros((10, ))
    vec[current_action] = 1
    current_action_vec = torch.from_numpy(vec).type(torch.float).view(-1, 10)
    next_action_prob = F.softmax(pred_model.model(None, data, current_action_vec, eval=True), dim=-1)
    print(next_action_prob)


if __name__ == "__main__":
    transition_server()
    # test()