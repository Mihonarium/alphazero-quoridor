import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS

import ray

from utils import *

class selfPlay():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.nnet.init_model()
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = dotdict(args)
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def play(self):
        # Execute one episode of self play
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            if np.sum(pi) == 0: break
            # sym = self.game.getSymmetries(canonicalBoard, pi)
            # for b, p in sym:
            #    trainExamples.append([b, self.curPlayer, p, None])
            
            action = np.random.choice(len(pi), p=pi)
            trainExamples.append([canonicalBoard, self.curPlayer, pi, None])
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
