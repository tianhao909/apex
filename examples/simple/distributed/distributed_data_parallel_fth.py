import torch  # 导入 PyTorch 库
import argparse  # 导入 argparse 库，用于处理命令行参数
import os  # 导入 os 库，用于操作系统相关功能
from apex import amp  # 导入 Apex 库的 amp 模块，用于混合精度训练
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel  # 导入 Apex 库的 DistributedDataParallel 模块，用于分布式训练

parser = argparse.ArgumentParser()  # 创建命令行解析器
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)  # 添加命令行参数 --local_rank，表示当前进程的本地编号，默认值为0
args = parser.parse_args()  # 解析命令行参数

# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
args.distributed = False  # 默认为非分布式训练
if 'WORLD_SIZE' in os.environ:  # 如果环境变量 WORLD_SIZE 存在
    args.distributed = int(os.environ['WORLD_SIZE']) > 1  # 根据环境变量判断是否为分布式训练

if args.distributed:  # 如果是分布式训练
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)  # 根据 local_rank 设置当前设备

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',  # 初始化分布式训练进程组，使用 NCCL 后端
                                         init_method='env://')  # 使用环境变量初始化

torch.backends.cudnn.benchmark = True  # 启用 cudnn 基准测试，以便为每个配置选择最佳算法

N, D_in, D_out = 64, 1024, 16  # 设置批次大小、输入维度和输出维度

# Each process receives its own batch of "fake input data" and "fake target data."
# The "training loop" in each process just uses this fake batch over and over.
# https://github.com/NVIDIA/apex/tree/master/examples/imagenet provides a more realistic
# example of distributed data sampling for both training and validation.
x = torch.randn(N, D_in, device='cuda')  # 生成一个随机的输入数据张量 x
y = torch.randn(N, D_out, device='cuda')  # 生成一个随机的目标数据张量 y

model = torch.nn.Linear(D_in, D_out).cuda()  # 创建一个线性层模型并将其移动到 GPU 上
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 使用 SGD 优化器，学习率为 1e-3

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 初始化模型和优化器，启用混合精度训练，opt_level="O1"

if args.distributed:  # 如果是分布式训练
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    # apex.parallel.DistributedDataParallel.
    model = DistributedDataParallel(model)  # 使用 Apex 的 DistributedDataParallel 包装模型
    # torch.nn.parallel.DistributedDataParallel is also fine, with some added args:
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank)

loss_fn = torch.nn.MSELoss()  # 创建均方误差损失函数

for t in range(500):  # 训练循环，迭代 500 次
    optimizer.zero_grad()  # 清空梯度
    y_pred = model(x)  # 使用模型进行前向传播
    loss = loss_fn(y_pred, y)  # 计算损失
    with amp.scale_loss(loss, optimizer) as scaled_loss:  # 使用 amp.scale_loss 进行损失缩放
        scaled_loss.backward()  # 反向传播
    optimizer.step()  # 更新优化器

if args.local_rank == 0:  # 如果是进程 0（主进程）
    print("final loss = ", loss)  # 打印最终损失
