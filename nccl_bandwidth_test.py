"""
GPU 间 NCCL P2P 带宽测试 (使用 torch.distributed)

这个脚本通过 torch.distributed 的 NCCL 后端做 send/recv,
NCCL_P2P_DISABLE 才会真正生效。

用法:
    # 启用 P2P (默认)
    torchrun --nproc_per_node=2 nccl_bandwidth_test.py 2>&1 | tee bw_p2p_on.log

    # 禁用 P2P
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 nccl_bandwidth_test.py 2>&1 | tee bw_p2p_off.log

    # 想看 NCCL 实际选了哪条路径,加上 NCCL_DEBUG:
    NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 nccl_bandwidth_test.py
"""

import os
import time
import torch
import torch.distributed as dist


def measure_bandwidth(rank: int, size_mb: int,
                      n_warmup: int = 5, n_iter: int = 20):
    """
    rank 0 发送, rank 1 接收
    返回 rank 0 测得的带宽数据
    """
    n_elements = size_mb * 1024 * 1024 // 4  # float32
    num_bytes = n_elements * 4

    if rank == 0:
        tensor = torch.randn(n_elements, dtype=torch.float32, device='cuda')
    else:
        tensor = torch.empty(n_elements, dtype=torch.float32, device='cuda')

    # warmup
    for _ in range(n_warmup):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
    torch.cuda.synchronize()
    dist.barrier()

    # 正式测量
    timings = []
    for _ in range(n_iter):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    dist.barrier()

    if rank == 0:
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        return {
            'size_mb': size_mb,
            'avg_ms': avg_time * 1000,
            'min_ms': min_time * 1000,
            'bw_avg_gbps': num_bytes / avg_time / 1e9,
            'bw_peak_gbps': num_bytes / min_time / 1e9,
        }
    return None


def main():
    # torchrun 会自动设置这些环境变量
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    assert world_size == 2, "这个脚本需要 2 个进程 (--nproc_per_node=2)"

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    if rank == 0:
        print("=" * 70)
        print("NCCL send/recv 带宽测试")
        print("=" * 70)
        print(f"NCCL_P2P_DISABLE = {os.environ.get('NCCL_P2P_DISABLE', '未设置 (P2P 启用)')}")
        print(f"NCCL_MAX_NCHANNELS = {os.environ.get('NCCL_MAX_NCHANNELS', '未设置 (默认)')}")
        print(f"World size: {world_size}")
        print(f"Rank 0: GPU{local_rank} ({torch.cuda.get_device_name(local_rank)})")
        print()
        print(f"{'Size':>8} | {'Avg Time':>10} | {'Min Time':>10} | "
              f"{'Avg BW':>12} | {'Peak BW':>12}")
        print("-" * 70)

    sizes_mb = [1, 4, 16, 64, 256, 1024]

    for size_mb in sizes_mb:
        try:
            result = measure_bandwidth(rank, size_mb)
            if rank == 0 and result:
                print(f"{result['size_mb']:>5} MB | "
                      f"{result['avg_ms']:>8.2f} ms | "
                      f"{result['min_ms']:>8.2f} ms | "
                      f"{result['bw_avg_gbps']:>8.2f} GB/s | "
                      f"{result['bw_peak_gbps']:>8.2f} GB/s")
        except RuntimeError as e:
            if rank == 0:
                print(f"{size_mb:>5} MB | 失败: {e}")
            break

    if rank == 0:
        print()
        print("看 256/1024 MB 的 Avg BW,这是稳态带宽,直接对应你迁移 KV cache 的场景。")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()