
import ray
import os
import torch
import torch.distributed as dist


WORLD_SIZE = 2


@ray.remote(num_gpus=1)
def test_nccl(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '27000'

    print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    dist.init_process_group(backend='nccl', world_size=WORLD_SIZE, rank=rank)

    if rank == 0:
        data = torch.rand(3, 3).cuda()
        dist.broadcast(data, 0)
        print(f'Sending data from rank 0: {data}')
    else:
        data = torch.zeros(3, 3).cuda()
        dist.broadcast(data, 0)
        print(f'Received data on rank {rank}: {data}')


refs = []

for rank in range(WORLD_SIZE):
    refs.append(test_nccl.remote(rank))

ray.get(refs)
