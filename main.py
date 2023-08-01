import logging
from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame as Game
from quoridor.pytorch.NNet import NNetWrapper as nn
from utils import *
import ray
ray.init()

log = logging.getLogger(__name__)

raw_args = {
    'numIters': 50, #1000
    'numEps': 50, #100
    'tempThreshold': 25,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','checkpoint_5.pth.tar'),
    'numItersForTrainExamplesHistory': 50,
}

args_ref = ray.put(raw_args)

if __name__=="__main__":
    args = dotdict(raw_args)
    log.info('Loading %s...', Game.__name__)
    g = Game(5)
    print(g.getBoardSize())
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args_ref)
    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()
    log.info('Starting the learning process 🎉')
    c.learn()
