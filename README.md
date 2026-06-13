<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM Migration Experiments

这个仓库基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 扩展了一套 KV cache 迁移实验框架。原始 nano-vLLM 是一个轻量级离线推理引擎；当前 `migrateV1` 分支重点研究的是：当一个请求已经在源 GPU 或源机器上完成 prefill 后，如何把它的推理状态迁移到目标 GPU 或目标机器，并比较不同迁移策略的耗时。

当前分支的核心代码集中在两个文件：

- `nanovllm/engine/migrator.py`: 迁移执行器，负责 prefill、KV cache 打包/传输/写入、目标端重算、decode 阶段迁移和 baseline 策略。
- `benchmarks/hybrid_migration_bench.py`: 实验入口，负责启动两个 NCCL rank、构造模型 runner、验证正确性、运行 Exp1/Exp2/Exp3、保存 JSON 和图。

辅助脚本：

- `nccl_bandwidth_test.py`: 测 GPU 间 NCCL `send/recv` 带宽，用于理解 KV 迁移的通信瓶颈。
- `example.py`: 原 nano-vLLM 的基础文本生成示例。
- `bench.py`: 原 nano-vLLM 的离线吞吐 benchmark。

## Background

LLM 推理中的请求状态主要包括 token 序列和 KV cache。KV cache 体积远大于 token id，但它能避免目标端重新 prefill；token id 很小，但目标端收到后必须重新计算 KV。

本项目把迁移策略抽象成三个端点和一个中间区域：

- `alpha = 0`: 纯 KV migration。目标端直接接收全部 KV cache，几乎不重算 prompt。
- `alpha = 1`: 纯 token migration / recompute。目标端只依赖 token ids，重新 prefill 全部 prompt。
- `0 < alpha < 1`: hybrid migration。目标端重算前 `alpha` 比例的 token，同时接收后半部分 KV cache，让计算和通信重叠。

代码会把 `alpha` 对齐到 KV block 边界，因为 nano-vLLM 的 KV cache 按 block 管理。

## Implemented Features

### 1. Standalone ModelRunner

原始 nano-vLLM 的 `ModelRunner` 默认会自己初始化 tensor parallel 的 NCCL process group。迁移实验需要外层 benchmark 脚本自己管理两个 rank，所以当前分支为 `ModelRunner` 增加了 `standalone=True` 模式：

- 跳过内部 `dist.init_process_group`。
- 每个 rank 加载一份完整模型，而不是 tensor parallel shard。
- 强制 eager execution，避免 CUDA graph 对实验时序造成干扰。
- 保留 nano-vLLM 原本的 prefill、decode、KV cache 分配逻辑。

benchmark 中会临时 monkey patch `dist.get_world_size/get_rank`，让模型层认为当前 world size 是 1，从而在两个 rank 上都加载完整模型。

### 2. Hybrid KV Migration

`MigrationEngine.migrate(token_ids, alpha, src_seq)` 实现 prefill 后的混合迁移。

实验流程：

1. rank 0 作为源端，对输入 token 做 prefill，并在源端 KV cache 中留下完整状态。
2. 根据 `alpha` 计算 split point。
3. split point 前面的 token 由 rank 1 目标端重新 prefill。
4. split point 后面的 KV block 由 rank 0 打包，通过 NCCL 发送给 rank 1。
5. rank 1 在独立 CUDA stream 上执行重算，同时异步接收 KV。
6. rank 1 把收到的 KV 写入自己的 KV cache，形成可继续 decode 的 `Sequence`。
7. correctness check 会在迁移后分别从源端和目标端 greedy decode，并比较生成 token 是否一致。

这里的总耗时近似为：

```text
T_total(alpha, N) = max(T_recompute(alpha * N), T_transfer((1 - alpha) * N))
```

由于通信和重算可以重叠，最佳 `alpha` 通常不是 0 或 1，而是在二者耗时接近平衡的位置。

### 3. Bandwidth Scaling

`--bandwidth-scale K` 用来模拟更低的链路带宽。当前实现不会把 buffer 真的扩成 K 倍，而是发送同一个 KV buffer K 次：

- 第一次是真实 KV 数据。
- 后面 `K - 1` 次是 padding transfer，用来增加通信耗时。
- 这样可以模拟慢链路，同时避免因为构造巨大 buffer 造成显存 OOM。

例如 `--bandwidth-scale 4` 近似模拟只有当前实测带宽四分之一的传输环境。

### 4. Profiling-Based Alpha Prediction

开启 `--profile` 后，benchmark 会先测量两条曲线：

- `T_transfer(N)`: `alpha=0`，纯 KV 传输耗时。
- `T_compute(N)`: `alpha=1`，目标端纯重算耗时。

然后拟合：

```text
T_compute(n)  = c0 + c1 * n + c2 * n^2
T_transfer(n) = t0 + t1 * n
```

接着求解：

```text
T_compute(alpha * N) = T_transfer((1 - alpha) * N)
```

得到每个目标序列长度 `N` 对应的预测最优 `alpha*(N)`。预测值会再对齐到 KV block 边界。实验结果里也会保留 brute-force sweep 得到的最优 `alpha`，方便比较预测是否准确。

### 5. Decode-Phase Migration

`migrateV1` 分支还实现了 decode 阶段迁移实验，也就是请求已经在源端开始 decode 时，目标端如何接管。

当前 Exp3 比较三种策略：

- `token_migration`: 源端发送 token ids，目标端重新 prefill，然后目标端 decode。
- `deferred_kv`: 源端不在迁移期间继续做有用 decode，只发送已有完整 KV，目标端收到后继续 decode。
- `iterative_kv`: 源端一边分轮发送尚未迁移的 KV，一边继续 decode；目标端每轮接收 KV 并追赶状态；最后目标端 decode 剩余 token。

`iterative_kv` 的一轮大致是：

1. 源端计算还有多少 token 的 KV 未迁移。
2. 源端按 block 打包未迁移范围内的 KV。
3. 源端异步发送 KV。
4. 在等待目标端完成本轮接收期间，源端继续 greedy decode。
5. 目标端接收 KV，写入自己的 KV cache。
6. 目标端发送 done 信号。
7. 源端把本轮新 decode 出来的 token ids 发给目标端。
8. 目标端扩展自己的 `Sequence` 和 block table。
9. 重复直到目标端追上当前状态，然后目标端 decode 剩余 token。

注意：代码注释中仍有“prefill N, then decode N”的描述，但当前 `run_exp3` 实际把 `max_decode` 固定为 `1024`。因此当前 Exp3 是“prefill N，然后 decode 1024 个 token”，不是每个 N 都 decode N 个 token。

## Experiments

### Exp1: Alpha Sweep

目标：在固定序列长度 `N` 下，观察不同 `alpha` 的迁移耗时。

方法：

- `alpha` 从 `0.00` 到 `1.00`，步长 `0.05`。
- 每个点先 warmup，再重复运行 `--num-repeats` 次。
- 记录 median/min/max。
- 输出 `exp1_alpha_sweep.png`。

这个实验回答的问题是：

```text
在某个 N 下，KV 传输、目标端重算、hybrid overlap 之间的最佳平衡点在哪里？
```

### Exp2: N Sweep

目标：比较不同序列长度下，各策略的迁移耗时增长趋势。

默认测试：

```text
N = 1024, 2048, 4096, 8192, 16384
```

策略：

- `kv_migration`: `alpha=0`
- `token_migration`: `alpha=1`
- `hybrid`: 使用 Exp1 找到的固定 `alpha*`，或者 `--profile` 预测出的 per-N `alpha*(N)`
- `hybrid_bruteforce`: 对每个 N 暴力 sweep `alpha` 得到的最优点

输出：

- `exp2_n_sweep.png`
- `results.json` 中的 `exp2`
- 如果开启 `--profile`，还会保存拟合参数和 `alpha_map`

这个实验回答的问题是：

```text
随着 prompt 变长，纯 KV、纯重算、hybrid 和理论最优 hybrid 的差距如何变化？
```

### Exp3: Decode-Phase Migration

目标：比较迁移发生在 decode 阶段时，三种接管策略的端到端迁移耗时。

默认测试：

```text
N = 1024, 2048, 4096, 8192, 16384
max_decode = 1024
```

策略：

- `token_migration`
- `deferred_kv`
- `iterative_kv`

输出：

- `exp3_decode_migration.png`
- `results.json` 中的 `exp3`
- 每个点还会记录 `num_rounds`, `src_decoded`, `dst_decoded`

这个实验回答的问题是：

```text
如果源端在迁移过程中继续 decode，iterative KV catch-up 是否能减少目标端接管总时间？
```

## Installation

建议使用 Python 3.10-3.12，并准备 CUDA、NCCL、PyTorch、Triton 和 FlashAttention 环境。

```bash
git clone https://github.com/YMbmzy/nano-vllm.git
cd nano-vllm
git checkout migrateV1
pip install -e .
```

`flash-attn` 对 CUDA/PyTorch 版本比较敏感。如果 `pip install -e .` 在安装 `flash-attn` 时失败，先根据你的 CUDA 和 PyTorch 版本安装匹配的 FlashAttention wheel，再重新安装本项目。

## Model Download

原始 demo 使用 Qwen3-0.6B：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ./Qwen3-0.6B \
  --local-dir-use-symlinks False
```

迁移实验通常使用更大的 Qwen3 模型路径，例如：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-4B \
  --local-dir ./Qwen3-4B \
  --local-dir-use-symlinks False
```

实际模型路径通过 `--model` 传入 benchmark。

## Quick Start

基础 nano-vLLM 生成示例：

```bash
python example.py
```

基础吞吐 benchmark：

```bash
python bench.py
```

## Run Migration Benchmarks

### Single Machine, 2 GPUs

运行 Exp1 和 Exp2：

```bash
python benchmarks/hybrid_migration_bench.py \
  --model ./Qwen3-4B \
  --block-size 256 \
  --bandwidth-scale 4 \
  --num-repeats 5 \
  --output-dir results
```

开启 profile，用拟合模型预测每个 N 的 `alpha*(N)`：

```bash
python benchmarks/hybrid_migration_bench.py \
  --model ./Qwen3-4B \
  --block-size 256 \
  --bandwidth-scale 4 \
  --num-repeats 5 \
  --profile \
  --output-dir results
```

同时运行 decode-phase Exp3：

```bash
python benchmarks/hybrid_migration_bench.py \
  --model ./Qwen3-4B \
  --block-size 256 \
  --bandwidth-scale 4 \
  --num-repeats 5 \
  --profile \
  --iterative \
  --output-dir results
```

### Cross Machine, 1 GPU Each

机器 0 作为源端：

```bash
python benchmarks/hybrid_migration_bench.py \
  --model ./Qwen3-4B \
  --rank 0 \
  --master-addr <MACHINE_0_IP> \
  --master-port 29500 \
  --block-size 256 \
  --bandwidth-scale 4 \
  --num-repeats 5 \
  --profile \
  --iterative \
  --output-dir results
```

机器 1 作为目标端：

```bash
python benchmarks/hybrid_migration_bench.py \
  --model ./Qwen3-4B \
  --rank 1 \
  --master-addr <MACHINE_0_IP> \
  --master-port 29500 \
  --block-size 256 \
  --bandwidth-scale 4 \
  --num-repeats 5 \
  --profile \
  --iterative \
  --output-dir results
```

跨机运行时，两台机器需要能互相访问 `MASTER_ADDR:MASTER_PORT`，并且 NCCL 环境变量要和你的网络设备匹配。

## Command Options

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | required | Hugging Face 模型目录 |
| `--block-size` | `256` | KV cache block size |
| `--num-repeats` | `5` | 每个数据点的重复次数 |
| `--exp1-n` | auto | 固定 Exp1 的序列长度；不传则自动选择 transfer 和 recompute 耗时接近的 N |
| `--output-dir` | `results` | 保存 JSON 和图的目录 |
| `--master-addr` | `localhost` | NCCL master address |
| `--master-port` | `29500` | NCCL master port |
| `--rank` | none | 跨机器模式手动指定 rank；不传则单机 `mp.spawn` 两个进程 |
| `--bandwidth-scale` | `1` | 通信耗时放大倍数，用于模拟低带宽 |
| `--profile` | off | 开启 latency profiling 和 per-N `alpha*(N)` 预测 |
| `--no-profile` | off | 显式关闭 profile |
| `--iterative` | off | 开启 Exp3 decode-phase migration |

## Outputs

默认输出目录是 `results/`：

```text
results/
  results.json
  exp1_alpha_sweep.png
  exp2_n_sweep.png
  exp3_decode_migration.png  # 只有开启 --iterative 时生成
```

`results.json` 主要字段：

- `exp1`: alpha sweep 的所有数据点。
- `exp2`: N sweep 的所有策略数据点。
- `exp3`: decode-phase migration 数据点，仅 `--iterative` 时存在。
- `chosen_N`: Exp1 使用的序列长度。
- `alpha_star`: Exp1 sweep 得到的最优 alpha，或 profile 模式下的 exp1 brute-force alpha。
- `profile`: profile 模式下的拟合参数、`alpha_map` 和预测信息。

## NCCL Bandwidth Test

单独测 GPU 间 send/recv 带宽：

```bash
torchrun --nproc_per_node=2 nccl_bandwidth_test.py
```

禁用 P2P 后再测：

```bash
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 nccl_bandwidth_test.py
```

如果要确认 NCCL 选了什么通信路径：

```bash
NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 nccl_bandwidth_test.py
```

重点看 256 MB 和 1024 MB 的 average bandwidth，它们更接近大块 KV cache 迁移时的稳态带宽。

## Reading Guide

如果你刚接触这个项目，可以按这个顺序读：

1. `benchmarks/hybrid_migration_bench.py`
   先看 `main()` 和 `worker_fn()`，理解两个 rank 如何启动、如何创建模型、如何组织实验。

2. `nanovllm/engine/migrator.py`
   先看 `MigrationEngine.migrate()`，理解 prefill 后 hybrid KV migration。

3. `benchmarks/hybrid_migration_bench.py`
   再看 `run_exp1()`, `run_exp2()`, `run_profiling()`, `predict_alpha_star()`，理解 `alpha` sweep 和 profile-based prediction。

4. `nanovllm/engine/migrator.py`
   最后看 `migrate_token_migration()`, `migrate_deferred_kv()`, `migrate_iterative()`，理解 decode-phase Exp3。

5. `nanovllm/engine/model_runner.py`
   需要理解 nano-vLLM 如何管理 KV cache 时，再看 `prepare_prefill()`, `prepare_decode()`, `allocate_kv_cache()`。

## Current Limitations

- 当前实验默认只支持两个 rank：rank 0 是源端，rank 1 是目标端。
- benchmark 假设每个 rank 加载完整模型，不测试 tensor parallel 分片迁移。
- `ModelRunner` 在迁移实验中使用 eager 模式，不评估 CUDA graph 对迁移场景的影响。
- `--bandwidth-scale` 是通过重复传输同一 buffer 模拟低带宽，不等价于真实网络拥塞或跨机拓扑。
- Exp3 当前固定 `max_decode = 1024`，代码注释中的“decode N”不是当前实际行为。
- correctness check 使用 greedy decode 对齐 token，主要验证 KV 状态是否一致，不覆盖采样随机性。

## License

This project inherits the MIT license from nano-vLLM. See `LICENSE` for details.
