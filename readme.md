# TromisNet

A deep reinforcement learning agent to play Tromis, a simplified version of two-player Tetris. The model learns exclusively through self-play.

Neural network built with Pytorch, using a C++ implementation of Tromis.

A blog post explaining an early version of this, which used the same DNN structure, can be found [here](https://paulkra.wordpress.com/2021/09/28/tromisnet-and-implementing-a2c/).

Since then, with hyper-parameter tuning, off-policy training with V-trace and other methods, the agent improved significantly. 

Currently it achieves a 99.7% winrate, so new benchmarks are needed to measure its performance.

The source contains 3 different python trainers. The best one, experimentally, is trainer_self.py which optimises GPU utilisation by doing off-policy rollouts during game simulations.