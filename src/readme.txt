To train the network you need:

CUDA drivers and a CUDA compatible GPU.
Python 3.X.X
Pytorch

Pytorch can be installed with pip with 'python -m pip install torch'

Begin training with the following command:

'python trainer.py [proc_count] [batch_size] [save_replays]'

proc_count should be around one less than the number of threads on your CPU.

batch_size should be around 1000 but if you notice that batches are processing too fast you can increase it.

save_replays should 0 during training. If you want to see game replays saved to disk change it to 1.

Loss values are saved to './losses.txt'. Lower loss is better.

The commandline interface will also tell you periodically how well the network is performing as a win/loss ratio.

