from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame as Game
from quoridor.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 500, #1000
    'numEps': 50, #100
    'tempThreshold': 0.3,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 300,
    'arenaCompare': 20,
    'cpuct': 2.4,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    'load_folder_examples_file': ('./temp','checkpoint_34.pth.tar'),
    'load_examples': False,
    'numItersForTrainExamplesHistory': 100,
})

if __name__=="__main__":
    g = Game(5)
    print(g.getBoardSize())
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_examples:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
