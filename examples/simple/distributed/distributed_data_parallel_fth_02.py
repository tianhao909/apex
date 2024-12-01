import torch
import argparse
import os
from apex import amp
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel

# FOR DISTRIBUTED: Get local_rank from environment variable set by torchrun
local_rank = int(os.environ['LOCAL_RANK'])  # 从环境变量中获取 local_rank

# Set the device according to local_rank
torch.cuda.set_device(local_rank)

# Initialize distributed training environment
torch.distributed.init_process_group(backend='nccl', init_method='env://')

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

# Each process receives its own batch of "fake input data" and "fake target data."
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Initialize mixed precision training
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if torch.distributed.is_initialized():
    # Wrap the model with DistributedDataParallel after amp.initialize
    model = DistributedDataParallel(model)

loss_fn = torch.nn.MSELoss()

for t in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()

# Only print final loss from rank 0
if local_rank == 0:
    print("final loss = ", loss)
