#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 distributed_data_parallel.py


torchrun --nproc_per_node=2 distributed_data_parallel.py
torchrun --nproc_per_node=2 /mnt/fth/software4/apex/examples/simple/distributed/distributed_data_parallel_fth_02.py
