# AlphaZero, Quoridor Version!

Based on the framework provided [here](https://github.com/suragnair/alpha-zero-general).

To start training, modify parameters in `main.py` and then start using

```
python main.py
```

A checkpoint after >8k games of self-play is available [here](https://huggingface.co/mishasamin/alphazero-quoridor/blob/main/checkpoint_164.pth.tar); examples of games against it are available [here](https://github.com/Mihonarium/alphazero-quoridor/wiki/Game-examples).

### Playing against it

![quoridor](https://github.com/xphoniex/alphazero-quoridor/raw/master/quoridor/output.gif)

Once you're done training, you need to modify `pit.py` to create one NN player, pointing it to your `best.pth.tar` and a human player.


During the game, you have a choice of ten actions:
* `u` (up)
* `d` (down)
* `r` (right)
* `l` (left)
* plus four diagonal move `ur`, `ul`, `dr`, `dl`

In order to place walls, you type `h` (for horizontal wall) or `v` (for vertical wall), press enter followed by `x y` of where you want the wall to be placed.
