import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from basic_model import BasicModel

class encoder(nn.Module):
	def __init__(self, args):
		super(encoder, self).__init__()
		self.fc1 = nn.Linear(args.in_dim, 64)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(64, 16)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(16, args.embed_dim)

	def forward(self, x):
		x = self.relu1(self.fc1(x))
		x = self.relu2(self.fc2(x))
		embedding = self.fc3(x)
		return embedding

class decoder(nn.Module):
	def __init__(self, args):
		super(decoder, self).__init__()
		self.fc1 = nn.Linear(args.embed_dim, 16)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(16, 64)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(64, args.in_dim)

	def forward(self, embedding):
		x = self.relu1(self.fc1(embedding))
		x = self.relu2(self.fc2(x))
		x_recon = self.fc3(x)
		return x_recon

class robot2human_mapping(nn.Module):
	def __init__(self, args):
		super(robot2human_mapping, self).__init__()
		self.fc_force = nn.Linear(args.robot_dim-1, 128)
		self.force_relu = nn.ReLU()
		self.fc_gap = nn.Linear(1, 128)
		self.gap_relu = nn.ReLU()
		self.fc_mapping = nn.Linear(256, args.embed_dim)

		# self.fc1 = nn.Linear(args.robot_dim, 256)
		# self.relu1 = nn.ReLU()
		# self.fc2 = nn.Linear(256, args.embed_dim)

	def forward(self, x):
		force_mapping = self.force_relu(self.fc_force(x[:, :-1]))
		gap_mapping = self.gap_relu(self.fc_gap(x[:, -1].view(-1, 1)))
		robot2human_embedding = self.fc_mapping(torch.cat((force_mapping, gap_mapping), dim=1))

		return robot2human_embedding

		# x = self.relu1(self.fc1(x))
		# robot2human_embedding = self.fc2(x)
		# return robot2human_embedding

class action_prediction(nn.Module):
	def __init__(self, args):
		super(action_prediction, self).__init__()
		self.fc11 = nn.Linear(args.embed_dim, 64)
		self.relu11 = nn.ReLU()
		self.fc12 = nn.Linear(args.action_dim, 64)
		self.relu12 = nn.ReLU()
		self.fc2 = nn.Linear(128, args.action_dim)

	def forward(self, embedding, current_action):
		x1 = self.relu11(self.fc11(embedding))
		x2 = self.relu12(self.fc12(current_action))
		next_action = self.fc2(torch.cat((x1, x2), dim=1))
		return next_action

class joint_transition_model(BasicModel):
	def __init__(self, args):
		super(joint_transition_model, self).__init__(args)
		self.human_encoder = encoder(args)
		self.human_decoder = decoder(args)
		self.action_predictor = action_prediction(args)
		self.robot_mapping = robot2human_mapping(args)
		self.recon_optimizer = optim.Adam(self.parameters(), lr=args.lr)
		self.pred_optimizer = optim.Adam(self.parameters(), lr=args.lr)
		self.use_cuda = args.cuda

	def forward(self, human_post, robot_post, current_action, eval=False):
		if not eval:
			human_embedding = self.human_encoder(human_post)
			robot2human_embedding = self.robot_mapping(robot_post)
			next_action_pred = self.action_predictor(robot2human_embedding, current_action)
			human_post_recon = self.human_decoder(human_embedding)
			return human_post_recon, next_action_pred, human_embedding, robot2human_embedding
		else:
			robot2human_embedding = self.robot_mapping(robot_post)
			next_action_pred = self.action_predictor(robot2human_embedding, current_action)
			return next_action_pred

	def compute_recon_loss(self, recon, human_post):
		recon_loss = F.mse_loss(recon, human_post)
		return recon_loss

	def compute_pred_loss(self, next_action_pred, next_action_target, robot2human_embedding, human_embedding):
		# print(next_action_pred)
		# print(next_action_target)
		# print(next_action_pred.size())
		# print(next_action_target.size())
		# exit()
		pred_loss = F.cross_entropy(next_action_pred, next_action_target)
		mapping_loss = F.mse_loss(robot2human_embedding, human_embedding.detach())
		total_pred_loss = pred_loss + mapping_loss
		return total_pred_loss

	def load_model(self, path):
		state_dict = torch.load(path+'model.pth')
		self.human_encoder.load_state_dict(state_dict['human_encoder'])
		self.human_decoder.load_state_dict(state_dict['human_decoder'])
		self.action_predictor.load_state_dict(state_dict['action_predictor'])
		self.robot_mapping.load_state_dict(state_dict['robot_mapping'])
		self.recon_optimizer.load_state_dict(state_dict['recon_optim'])
		self.pred_optimizer.load_state_dict(state_dict['pred_optim'])

	def save_model(self, path):
		state_dict = {}
		state_dict['human_encoder'] = self.human_encoder.state_dict()
		state_dict['human_decoder'] = self.human_decoder.state_dict()
		state_dict['action_predictor'] = self.action_predictor.state_dict()
		state_dict['robot_mapping'] = self.robot_mapping.state_dict()
		state_dict['recon_optim'] = self.recon_optimizer.state_dict()
		state_dict['pred_optim'] = self.pred_optimizer.state_dict()
		torch.save(state_dict, path+'model.pth')
