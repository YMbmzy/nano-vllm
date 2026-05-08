#!/usr/bin/env python3
"""Hybrid KV Cache Migration Benchmark.

Spawns 2 processes (rank 0 = src, rank 1 = dst) via NCCL.
Runs two experiments:
  Exp 1: α sweep at fixed N  →  T_total vs α
  Exp 2: N sweep at fixed α* →  T_total vs N  (3 strategies)

Usage (single-machine, 2 GPUs):
    python benchmarks/hybrid_migration_bench.py \
        --model ./Qwen3-4B \
        --block-size 256 \
        --num-repeats 5 \
        --output-dir results

Usage (cross-machine, 1 GPU each):
    # Machine 0 (src):
    python benchmarks/hybrid_migration_bench.py \
        --model ./Qwen3-4B \
        --rank 0 --master-addr <MACHINE_0_IP> --master-port 29500
    # Machine 1 (dst):
    python benchmarks/hybrid_migration_bench.py \
        --model ./Qwen3-4B \
        --rank 1 --master-addr <MACHINE_0_IP> --master-port 29500
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.migrator import MigrationEngine


# ====================================================================== #
#  ModelRunner creation with monkey-patched dist (full model per rank)
# ====================================================================== #

def create_runner(rank: int, config: Config) -> ModelRunner:
    """Create a ModelRunner that loads the FULL (unsharded) model.

    dist.init_process_group is already called with world_size=2.
    We temporarily patch dist.get_world_size/get_rank so that model layers
    store tp_size=1 and never shard or all_reduce.
    """
    orig_ws = dist.get_world_size
    orig_rk = dist.get_rank
    dist.get_world_size = lambda *a, **kw: 1
    dist.get_rank = lambda *a, **kw: 0
    try:
        runner = ModelRunner(config, rank, standalone=True)
    finally:
        dist.get_world_size = orig_ws
        dist.get_rank = orig_rk
    return runner


# ====================================================================== #
#  Correctness check helpers
# ====================================================================== #

def verify_migration(engine: MigrationEngine, token_ids: list[int],
                     alpha: float, num_decode: int = 20) -> bool:
    """Run one migration + greedy decode on both ranks, compare tokens.

    Returns True on rank 1 if tokens match; rank 0 always returns True.
    """
    # -- prefill on src --
    src_seq = None
    if engine.rank == 0:
        src_seq = engine.prefill_src(token_ids)

    # -- migrate --
    _, dst_seq = engine.migrate(token_ids, alpha, src_seq=src_seq)

    dist.barrier()

    # -- decode on both sides --
    if engine.rank == 0:
        baseline = engine.greedy_decode(engine.src, engine.src_bm, src_seq, num_decode)
        device = f"cuda:{engine.src.rank}"
        baseline_t = torch.tensor(baseline, dtype=torch.int64, device=device)
        dist.send(baseline_t, dst=1)
        engine.cleanup_src(src_seq)
        return True
    else:
        migrated = engine.greedy_decode(engine.dst, engine.dst_bm, dst_seq, num_decode)
        device = f"cuda:{engine.dst.rank}"
        baseline_t = torch.empty(num_decode, dtype=torch.int64, device=device)
        dist.recv(baseline_t, src=0)
        engine.cleanup_dst(dst_seq)
        ok = migrated == baseline_t.tolist()
        if not ok:
            print(f"  [rank 1] MISMATCH: baseline={baseline_t.tolist()[:5]}... "
                  f"migrated={migrated[:5]}...")
        return ok


# ====================================================================== #
#  Single migration run (timed, no decode)
# ====================================================================== #

def run_one(engine: MigrationEngine, token_ids: list[int], alpha: float) -> float:
    """Returns T_total (ms) on rank 1; 0.0 on rank 0."""
    src_seq = None
    if engine.rank == 0:
        src_seq = engine.prefill_src(token_ids)
    t, dst_seq = engine.migrate(token_ids, alpha, src_seq=src_seq)
    dist.barrier()
    if engine.rank == 0:
        engine.cleanup_src(src_seq)
    else:
        engine.cleanup_dst(dst_seq)
    return t


# ====================================================================== #
#  Experiment 1: α sweep
# ====================================================================== #

def run_exp1(engine: MigrationEngine, token_ids: list[int],
             num_repeats: int) -> list[dict]:
    alphas = [round(i * 0.1, 1) for i in range(11)]
    results = []
    N = len(token_ids)

    for alpha in alphas:
        # warmup
        run_one(engine, token_ids, alpha)

        times = []
        for _ in range(num_repeats):
            t = run_one(engine, token_ids, alpha)
            times.append(t)

        if engine.rank == 1:
            times.sort()
            actual_alpha = engine.compute_split(N, alpha) / N
            r = dict(alpha=alpha, actual_alpha=actual_alpha,
                     median=times[len(times) // 2],
                     min=min(times), max=max(times), all=times)
            results.append(r)
            print(f"  α={alpha:.1f} (actual={actual_alpha:.3f}): "
                  f"median={r['median']:.2f} ms  "
                  f"[{r['min']:.2f}, {r['max']:.2f}]")

    return results


# ====================================================================== #
#  Experiment 2: N sweep
# ====================================================================== #

def run_exp2(engine: MigrationEngine, prompt_tokens: list[int],
             alpha_star: float, num_repeats: int) -> list[dict]:
    Ns = [512, 1024, 2048, 4096]
    strategies = {"kv_migration": 0.0, "token_migration": 1.0, "hybrid": alpha_star}
    results = []

    for N in Ns:
        if N > len(prompt_tokens):
            if engine.rank == 1:
                print(f"  Skipping N={N}: prompt too short ({len(prompt_tokens)} tokens)")
            continue

        token_ids = prompt_tokens[:N]

        for name, alpha in strategies.items():
            # warmup
            run_one(engine, token_ids, alpha)

            times = []
            for _ in range(num_repeats):
                t = run_one(engine, token_ids, alpha)
                times.append(t)

            if engine.rank == 1:
                times.sort()
                r = dict(N=N, strategy=name, alpha=alpha,
                         median=times[len(times) // 2],
                         min=min(times), max=max(times), all=times)
                results.append(r)
                print(f"  N={N} {name:20s} (α={alpha:.2f}): "
                      f"median={r['median']:.2f} ms  "
                      f"[{r['min']:.2f}, {r['max']:.2f}]")

    return results


# ====================================================================== #
#  Calibration: find N where T_transfer ≈ T_recompute
# ====================================================================== #

def calibrate(engine: MigrationEngine, prompt_tokens: list[int]) -> int:
    """Test candidate Ns and pick the one where transfer and recompute
    times are closest. Returns the chosen N."""
    candidates = [512, 1024, 2048, 4096]
    best_N, best_ratio = 1024, float("inf")

    for N in candidates:
        if N > len(prompt_tokens):
            continue
        token_ids = prompt_tokens[:N]

        # pure transfer (α=0)
        t_transfer = run_one(engine, token_ids, alpha=0.0)
        # pure recompute (α=1)
        t_recompute = run_one(engine, token_ids, alpha=1.0)

        if engine.rank == 1 and t_transfer > 0 and t_recompute > 0:
            ratio = max(t_transfer, t_recompute) / min(t_transfer, t_recompute)
            print(f"  N={N}: T_transfer={t_transfer:.2f} ms, "
                  f"T_recompute={t_recompute:.2f} ms, ratio={ratio:.2f}")
            if ratio < best_ratio:
                best_ratio = ratio
                best_N = N

    # broadcast chosen N from rank 1 to rank 0
    runner = engine.src if engine.rank == 0 else engine.dst
    n_tensor = torch.tensor([best_N], dtype=torch.int64, device=f"cuda:{runner.rank}")
    dist.broadcast(n_tensor, src=1)
    return n_tensor.item()


# ====================================================================== #
#  Plotting
# ====================================================================== #

def plot_exp1(results: list[dict], N: int, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [r["actual_alpha"] for r in results]
    medians = [r["median"] for r in results]
    lo = [r["median"] - r["min"] for r in results]
    hi = [r["max"] - r["median"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(alphas, medians, yerr=[lo, hi], marker="o", capsize=3, linewidth=2)

    min_idx = medians.index(min(medians))
    ax.annotate("≈ KV Migration\n(Llumnix)", xy=(alphas[0], medians[0]),
                xytext=(alphas[0] + 0.08, medians[0] * 1.08),
                fontsize=9, ha="left")
    ax.annotate("≈ Token Migration\n(ServerlessLLM)", xy=(alphas[-1], medians[-1]),
                xytext=(alphas[-1] - 0.08, medians[-1] * 1.08),
                fontsize=9, ha="right")
    ax.annotate(f"Hybrid Optimum\n(α*≈{alphas[min_idx]:.2f})",
                xy=(alphas[min_idx], medians[min_idx]),
                xytext=(alphas[min_idx], medians[min_idx] * 0.82),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlabel("α (recompute fraction)", fontsize=12)
    ax.set_ylabel("T_total (ms)", fontsize=12)
    ax.set_title(f"T_total vs α  (N = {N})", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


def plot_exp2(results: list[dict], alpha_star: float, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"kv_migration": "KV Migration (α=0)",
              "token_migration": "Token Migration (α=1)",
              "hybrid": f"Hybrid (α={alpha_star:.2f})"}
    colors = {"kv_migration": "#1f77b4", "token_migration": "#ff7f0e", "hybrid": "#2ca02c"}
    markers = {"kv_migration": "s", "token_migration": "^", "hybrid": "o"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for strat in ["kv_migration", "token_migration", "hybrid"]:
        data = [r for r in results if r["strategy"] == strat]
        if not data:
            continue
        Ns = [r["N"] for r in data]
        med = [r["median"] for r in data]
        lo = [r["median"] - r["min"] for r in data]
        hi = [r["max"] - r["median"] for r in data]
        ax.errorbar(Ns, med, yerr=[lo, hi], marker=markers[strat], capsize=3,
                    linewidth=2, label=labels[strat], color=colors[strat])

    ax.set_xlabel("Sequence Length N", fontsize=12)
    ax.set_ylabel("T_total (ms)", fontsize=12)
    ax.set_title("T_total vs Sequence Length", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


# ====================================================================== #
#  Worker entry point (one per rank)
# ====================================================================== #

def worker_fn(rank: int, world_size: int, args, result_queue):
    # -- init NCCL --
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group("nccl", world_size=world_size, rank=rank)

    # cross-machine: each machine has 1 GPU (cuda:0)
    # single-machine: rank 0 → cuda:0, rank 1 → cuda:1
    gpu_id = 0 if args.rank is not None else rank

    if rank == 1:
        print(f"[rank {rank}] NCCL initialized, gpu_id={gpu_id}")

    # -- create runner (monkey-patched, full model) --
    config = Config(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        kvcache_block_size=args.block_size,
        max_model_len=8192,
    )
    Sequence.block_size = config.kvcache_block_size
    runner = create_runner(gpu_id, config)
    bm = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

    if rank == 1:
        print(f"[rank {rank}] ModelRunner ready, "
              f"KV blocks={config.num_kvcache_blocks}, block_size={args.block_size}")

    # -- build engine --
    src_runner = runner if rank == 0 else None
    dst_runner = runner if rank == 1 else None
    src_bm = bm if rank == 0 else None
    dst_bm = bm if rank == 1 else None
    engine = MigrationEngine(rank, src_runner, dst_runner, src_bm, dst_bm)

    # -- tokenize a long prompt (both ranks use the same tokens) --
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    base_text = "The quick brown fox jumps over the lazy dog. " * 500
    prompt_tokens = tokenizer.encode(base_text)[:8192]
    if rank == 1:
        print(f"Prompt tokens available: {len(prompt_tokens)}")

    # ============================================================ #
    #  Correctness check
    # ============================================================ #
    if rank == 1:
        print("\n=== Correctness check (α=0.5) ===")
    N_check = min(1024, len(prompt_tokens))
    ok = verify_migration(engine, prompt_tokens[:N_check], alpha=0.5)
    if rank == 1:
        print(f"  Result: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  Aborting due to correctness failure.")
            dist.destroy_process_group()
            return

    # ============================================================ #
    #  Calibration
    # ============================================================ #
    if rank == 1:
        print("\n=== Calibration ===")
    if args.exp1_n:
        chosen_N = args.exp1_n
        if rank == 1:
            print(f"  Using user-specified N={chosen_N}")
    else:
        chosen_N = calibrate(engine, prompt_tokens)
        if rank == 1:
            print(f"  Chosen N={chosen_N}")

    # ============================================================ #
    #  Experiment 1: α sweep
    # ============================================================ #
    if rank == 1:
        print(f"\n=== Experiment 1: α sweep at N={chosen_N} ===")
    exp1_tokens = prompt_tokens[:chosen_N]
    exp1_results = run_exp1(engine, exp1_tokens, args.num_repeats)

    # find α* from exp1
    alpha_star = 0.5
    if rank == 1 and exp1_results:
        best = min(exp1_results, key=lambda r: r["median"])
        alpha_star = best["alpha"]
        print(f"  → α* = {alpha_star:.1f}")

    # broadcast α* to rank 0
    a_tensor = torch.tensor([alpha_star], dtype=torch.float32, device=f"cuda:{gpu_id}")
    dist.broadcast(a_tensor, src=1)
    alpha_star = a_tensor.item()

    # ============================================================ #
    #  Experiment 2: N sweep
    # ============================================================ #
    if rank == 1:
        print(f"\n=== Experiment 2: N sweep at α*={alpha_star:.2f} ===")
    exp2_results = run_exp2(engine, prompt_tokens, alpha_star, args.num_repeats)

    # ============================================================ #
    #  Collect results
    # ============================================================ #
    if rank == 1:
        data = {
            "exp1": exp1_results,
            "exp2": exp2_results,
            "chosen_N": chosen_N,
            "alpha_star": alpha_star,
        }
        if result_queue is not None:
            result_queue.put(data)
        else:
            # cross-machine mode: save to file directly
            os.makedirs(args.output_dir, exist_ok=True)
            json_path = os.path.join(args.output_dir, "results.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nRaw results → {json_path}")
            if data["exp1"]:
                plot_exp1(data["exp1"], chosen_N,
                          os.path.join(args.output_dir, "exp1_alpha_sweep.png"))
            if data["exp2"]:
                plot_exp2(data["exp2"], alpha_star,
                          os.path.join(args.output_dir, "exp2_n_sweep.png"))

    dist.barrier()
    dist.destroy_process_group()


# ====================================================================== #
#  Main
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Hybrid Migration Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to Qwen3 model directory")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument("--exp1-n", type=int, default=None,
                        help="Fixed N for exp1. Auto-calibrate if omitted.")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--master-addr", type=str, default="localhost")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--rank", type=int, default=None,
                        help="Manual rank for cross-machine mode. "
                             "Omit for single-machine (mp.spawn).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.rank is not None:
        # cross-machine: each machine runs this script with its own --rank
        worker_fn(args.rank, 2, args, result_queue=None)
    else:
        # single-machine: spawn both workers locally
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        mp.spawn(worker_fn, args=(2, args, result_queue), nprocs=2, join=True)

        if not result_queue.empty():
            data = result_queue.get()
            json_path = os.path.join(args.output_dir, "results.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nRaw results → {json_path}")

            N = data["chosen_N"]
            alpha_star = data["alpha_star"]
            if data["exp1"]:
                plot_exp1(data["exp1"], N, os.path.join(args.output_dir, "exp1_alpha_sweep.png"))
            if data["exp2"]:
                plot_exp2(data["exp2"], alpha_star, os.path.join(args.output_dir, "exp2_n_sweep.png"))
        else:
            print("No results collected (queue empty).")


if __name__ == "__main__":
    main()
