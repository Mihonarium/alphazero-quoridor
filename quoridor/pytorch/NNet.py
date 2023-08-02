import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from .QuoridorNNet import QuoridorNNet as qnnet

args = dotdict({
    'lr': 0.002,
    'dropout': 0.3,
    'epochs': 3,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,
    'clip': 1.0,
    'weight_decay': 1e-4
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = qnnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.game = game

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples, withValids=True):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # wandb.init(project="quoridor alphazero", config=config_dict, mode="disabled" if IS_CI else "run")
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_time = time.time()
        if len(examples[0]) < 4:
            withValids = False

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                if withValids:
                    boards, pis, vs, valids = list(zip(*[examples[i] for i in sample_ids]))
                else:
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.uint8))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                if withValids:
                    valids = torch.FloatTensor(np.array(valids).astype(np.uint8))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                    if withValids:
                        valids = valids.contiguous().cuda()
                boards, target_pis, target_vs, valids = Variable(boards), Variable(target_pis), Variable(target_vs), Variable(valids)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards, withValids)
                if withValids:
                    out_pi = out_pi * valids
                    out_pi = F.softmax(out_pi, dim=1)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi * 0.02 + l_v

                # record loss
                #pi_losses.update(l_pi.data[0], boards.size(0))
                #v_losses.update(l_v.data[0], boards.size(0))
                pi_losses.update(l_pi.item(), boards.size(0)) # new
                v_losses.update(l_v.item(), boards.size(0)) # new

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), args.clip)
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, board, valids=None):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.uint8))
        if args.cuda: board = board.contiguous().cuda()
        #board = Variable(board, volatile=True)
        #boad = Variable(board, requires_grad=True)
        board = board.view(4, self.board_x, self.board_y)
        with torch.no_grad():   #new
            self.nnet.eval()
            pi, v = self.nnet(board, valids is not None)
            if valids is not None:
                pi = pi * torch.FloatTensor(valids.astype(np.uint8)).to(pi.device)
                pi = F.softmax(pi, dim=1)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location='cpu')
        self.nnet = self.nnet.to('cpu')
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.nnet.cuda()
