import os
import sys
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
from torchvision import transforms, utils

import transition_model
sys.path.append(os.path.join(os.getcwd(), '../'))

from DataSet import *

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--path', type=str, default='./data/')
parser.add_argument('--save', type=str, default='./results/checkpoint/')
parser.add_argument('--load', type=str, default='./results/checkpoint/')
parser.add_argument('--log', type=str, default='./results/log/')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--win_size', type=int, default=10)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.exists(args.log):
    os.makedirs(args.log)

_, train_loader = data_loader('../DataSet/data/train', '../DataSet/files_h_train', '../DataSet/files_r_train', 'dataset_train_save_tmp', args, bload=True)
_, test_loader = data_loader('../DataSet/data/test', '../DataSet/files_h_test', '../DataSet/files_r_test', 'dataset_test_save_tmp', args, bload=True)
args.in_dim = 80
args.robot_dim = 4
args.embed_dim = 8
args.action_dim = 10

model = transition_model.joint_transition_model(args)
if args.cuda:
    model = model.cuda()


def train(epoch):
    model.train()
    train_loss = 0
    accuracy = 0

    recon_loss_all = 0.0
    pred_loss_all = 0.0
    acc_all = 0.0
    counter = 0
    train_iter = iter(train_loader)
    for _ in tqdm(range(len(train_iter))):
        counter += 1
        post_mean_list_h, post_mean_list_r, current_label_list, next_label_list = next(train_iter)
        post_h = torch.cat(post_mean_list_h, dim=0).view(args.batch_size, -1)
        post_r = torch.cat(post_mean_list_r, dim=0).view(args.batch_size, -1)
        current_action = torch.cat(current_label_list).view(args.batch_size, -1)
        next_action = torch.LongTensor(next_label_list)

        # add noise
        # post_h += tdist.Normal(0.0, 0.1).sample(post_h.size())
        # post_r += tdist.Normal(0.0, 0.1).sample(post_r.size())

        if args.cuda:
            post_h = post_h.cuda()
            post_r = post_r.cuda()
            current_action = current_action.cuda()
            next_action = next_action.cuda()
        recon_loss, pred_loss, acc = model.train_(post_h, post_r, current_action, next_action)
        # print('Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc))
        recon_loss_all += recon_loss
        pred_loss_all += pred_loss
        acc_all += acc
    if counter > 0:
        print("Avg Training Loss: {:.6f}, {:.6f}, Acc: {:.4f}".format(recon_loss_all/float(counter), pred_loss/float(counter), acc_all/float(counter)))
    return recon_loss_all/float(counter), pred_loss/float(counter), acc_all/float(counter)

def test(epoch):
    model.eval()
    accuracy = 0

    acc_all = 0.0
    counter = 0
    test_iter = iter(test_loader)
    for _ in tqdm(range(len(test_iter))):
        counter += 1
        post_mean_list_h, post_mean_list_r, current_label_list, next_label_list = next(test_iter)
        post_h = torch.cat(post_mean_list_h, dim=0).view(args.batch_size, -1)
        post_r = torch.cat(post_mean_list_r, dim=0).view(args.batch_size, -1)
        current_action = torch.cat(current_label_list).view(args.batch_size, -1)
        next_action = torch.LongTensor(next_label_list)

        # if next_action[0] == 7.0:
        #     print('post mean:')
        #     print(post_r)
        #     print('current action:')
        #     print(current_action)
        #     print('next action:')
        #     print(next_action)

        if args.cuda:
            post_h = post_h.cuda()
            post_r = post_r.cuda()
            current_action = current_action.cuda()
            next_action = next_action.cuda()
        acc, next_action_prob = model.test_(post_h, post_r, current_action, next_action)
        # if next_action[0] ==7.0:
        #     print('next action prob:')
        #     print(next_action_prob)
        # print('Test: Epoch:{}, Batch:{}, Acc:{:.4f}.'.format(epoch, batch_idx, acc))  
        acc_all += acc
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all/float(counter)

def main():
    model.load_model(args.load)
    for epoch in range(0, args.epochs):
        print('*** epoch {} ***'.format(epoch))
        train(epoch)
        test(epoch)
        model.save_model(args.save)
        # loss = {'train':train_loss, 'val':val_loss}
        # acc = {'train':train_acc, 'val':val_acc, 'test':test_acc}
        # log.write_scalars('Loss', loss, epoch)
        # log.write_scalars('Accuracy', acc, epoch)
if __name__ == '__main__':
    # pass
    main()
