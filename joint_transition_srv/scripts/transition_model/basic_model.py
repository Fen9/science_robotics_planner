import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicModel(nn.Module):
	def __init__(self, args):
		super(BasicModel, self).__init__()
		pass	

	def load_model(self, path):
		self.state_dict = torch.load(path+'model.pth')

	def save_model(self, path):
		torch.save(self.state_dict(), path+'model.pth')
		# torch.save(self.state_dict(), path+'model.pth')

	def compute_recon_loss(self, recon, target):
		pass

	def compute_pred_loss(self, next_action_pred, next_action_target, robot2human_embedding, human_embedding):
		pass

	def train_(self, human_post, robot_post, current_action, next_action_target):
		self.recon_optimizer.zero_grad()
		self.pred_optimizer.zero_grad()
		output = self(human_post, robot_post, current_action)
		recon, next_action_pred, human_embedding, robot2human_embedding = output[0], output[1], output[2], output[3]
		
		recon_loss = self.compute_recon_loss(recon, human_post)
		recon_loss.backward()
		self.recon_optimizer.step()
		
		pred_loss = self.compute_pred_loss(next_action_pred, next_action_target, robot2human_embedding, human_embedding)
		pred_loss.backward()
		self.pred_optimizer.step()
		pred = next_action_pred.data.max(1)[1]
		correct = pred.eq(next_action_target.data).cpu().sum().numpy()
		accuracy = correct * 100. /next_action_target.size()[0]
		return recon_loss.item(), pred_loss.item(), accuracy

	def test_(self, human_post, robot_post, current_action, next_action_target):
		next_action_pred = self(None, robot_post, current_action, eval=True)
		pred = next_action_pred.data.max(1)[1]
		correct = pred.eq(next_action_target.data).cpu().sum().numpy()
		accuracy = correct * 100. /next_action_target.size()[0]
		return accuracy, F.softmax(next_action_pred, dim=-1)