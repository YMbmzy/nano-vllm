#!/usr/bin/env python3
"""Hybrid KV Cache Migration Benchmark.

Spawns 2 processes (rank 0 = src, rank 1 = dst) via NCCL.
Runs two experiments:
  Exp 1: α sweep at fixed N  →  T_total vs α
  Exp 2: N sweep at fixed α* →  T_total vs N  (3 strategies)

Usage (single-machine, 2 GPUs, --profile / --iterative is optional):
    python benchmarks/hybrid_migration_bench.py \
        --model ./Qwen3-4B \
        --block-size 256 \
        --bandwidth-scale 3 \
        --num-repeats 5 \
        --profile \
        --iterative \
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

import numpy as np
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
    torch.cuda.empty_cache()
    return t if engine.rank == 1 else 0.0


# ====================================================================== #
#  Brute-force α sweep for a single N (used by exp2)
# ====================================================================== #

def bruteforce_optimal(engine: MigrationEngine, token_ids: list[int],
                       num_repeats: int, alpha_step: float = 0.05) -> dict | None:
    """Sweep α at the given N, return the result dict for the best α.

    Only populated on rank 1; returns None on rank 0.
    """
    alphas = [round(i * alpha_step, 2) for i in range(int(1.0 / alpha_step) + 1)]
    N = len(token_ids)
    best = None

    for alpha in alphas:
        run_one(engine, token_ids, alpha)  # warmup
        times = [run_one(engine, token_ids, alpha) for _ in range(num_repeats)]

        if engine.rank == 1:
            times.sort()
            median = times[len(times) // 2]
            if best is None or median < best["median"]:
                actual_alpha = engine.compute_split(N, alpha) / N
                best = dict(N=N, strategy="hybrid_bruteforce", alpha=alpha,
                            actual_alpha=actual_alpha,
                            median=median, min=min(times), max=max(times),
                            all=times)

    return best


# ====================================================================== #
#  Experiment 1: α sweep
# ====================================================================== #

def run_exp1(engine: MigrationEngine, token_ids: list[int],
             num_repeats: int) -> list[dict]:
    alphas = [round(i * 0.05, 2) for i in range(21)]
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
            print(f"  α={alpha:.2f} (actual={actual_alpha:.3f}): "
                  f"median={r['median']:.2f} ms  "
                  f"[{r['min']:.2f}, {r['max']:.2f}]")

    return results


# ====================================================================== #
#  Experiment 2: N sweep
# ====================================================================== #

def run_exp2(engine: MigrationEngine, prompt_tokens: list[int],
             alpha_star: float, num_repeats: int) -> list[dict]:
    Ns = [1024, 2048, 4096, 8192, 8192 * 2]
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

        # brute-force optimal α for this N
        bf = bruteforce_optimal(engine, token_ids, num_repeats)
        if engine.rank == 1 and bf:
            results.append(bf)
            print(f"  N={N} {'hybrid_bruteforce':20s} (α={bf['alpha']:.2f}): "
                  f"median={bf['median']:.2f} ms  "
                  f"[{bf['min']:.2f}, {bf['max']:.2f}]")

    return results


# ====================================================================== #
#  Calibration: find N where T_transfer ≈ T_recompute
# ====================================================================== #

def calibrate(engine: MigrationEngine, prompt_tokens: list[int]) -> int:
    """Test candidate Ns and pick the one where transfer and recompute
    times are closest. Returns the chosen N."""
    candidates = [1024, 2048, 4096, 8192, 8192 * 2]
    best_N, best_ratio = 4096, float("inf")

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
#  Profiling: dense N sampling at α=0 and α=1
# ====================================================================== #

def run_profiling(engine: MigrationEngine, prompt_tokens: list[int],
                  num_repeats: int = 5, step: int = 1024) -> tuple[list, list]:
    """Measure T_transfer(N) and T_compute(N) at dense N grid.

    Returns (compute_data, transfer_data) — lists of (N, T_median_ms).
    Only populated on rank 1; empty on rank 0.
    """
    Ns = list(range(step, 8192 * 2 + 1, step))
    compute_data, transfer_data = [], []

    for N in Ns:
        if N > len(prompt_tokens):
            continue
        token_ids = prompt_tokens[:N]

        # warmup both
        run_one(engine, token_ids, alpha=0.0)
        run_one(engine, token_ids, alpha=1.0)

        # pure transfer (α=0)
        t_list = [run_one(engine, token_ids, alpha=0.0) for _ in range(num_repeats)]
        # pure compute  (α=1)
        c_list = [run_one(engine, token_ids, alpha=1.0) for _ in range(num_repeats)]

        if engine.rank == 1:
            t_list.sort(); c_list.sort()
            t_med = t_list[len(t_list) // 2]
            c_med = c_list[len(c_list) // 2]
            transfer_data.append((N, t_med))
            compute_data.append((N, c_med))
            print(f"  N={N:5d}: T_transfer={t_med:7.2f} ms, T_compute={c_med:7.2f} ms")

    return compute_data, transfer_data


# ====================================================================== #
#  Model fitting
# ====================================================================== #

def fit_latency_models(compute_data: list[tuple], transfer_data: list[tuple]):
    """Fit T_compute(n) = c0 + c1*n + c2*n^2,  T_transfer(n) = t0 + t1*n.

    If c2 < 0 (physically impossible — attention cost cannot be subquadratic),
    falls back to linear fit (c2 = 0).

    Returns (c_params, t_params) = ((c0, c1, c2), (t0, t1)).
    """
    # compute: quadratic
    Ns = np.array([d[0] for d in compute_data], dtype=np.float64)
    Ts = np.array([d[1] for d in compute_data], dtype=np.float64)
    A = np.column_stack([np.ones_like(Ns), Ns, Ns ** 2])
    c_params, *_ = np.linalg.lstsq(A, Ts, rcond=None)

    if c_params[2] < 0:
        print(f"  WARNING: fitted c₂={c_params[2]:.2e} < 0, falling back to linear fit")
        A_lin = np.column_stack([np.ones_like(Ns), Ns])
        c_lin, *_ = np.linalg.lstsq(A_lin, Ts, rcond=None)
        c_params = np.array([c_lin[0], c_lin[1], 0.0])

    # transfer: linear
    Ns = np.array([d[0] for d in transfer_data], dtype=np.float64)
    Ts = np.array([d[1] for d in transfer_data], dtype=np.float64)
    A = np.column_stack([np.ones_like(Ns), Ns])
    t_params, *_ = np.linalg.lstsq(A, Ts, rcond=None)

    return tuple(c_params), tuple(t_params)


# ====================================================================== #
#  Predict optimal α*(N) by solving the quadratic equation
# ====================================================================== #

def predict_alpha_star(N: int, block_size: int,
                       c_params: tuple, t_params: tuple) -> float:
    """Solve T_compute(αN) = T_transfer((1-α)N) for α, snap to block boundary.

    The equation:  c₂N²·α² + (c₁+t₁)N·α + (c₀ - t₀ - t₁N) = 0
    Take the root in [0, 1]; if none exists, pick the better boundary (0 or 1).
    Finally snap to the nearest block-aligned α.
    """
    c0, c1, c2 = c_params
    t0, t1 = t_params

    # boundary costs
    t_all_transfer = t0 + t1 * N            # α=0
    t_all_compute  = c0 + c1 * N + c2 * N * N  # α=1

    # quadratic coefficients: a·α² + b·α + c = 0
    qa = c2 * N * N
    qb = (c1 + t1) * N
    qc = c0 - t0 - t1 * N

    def _t_total_at(a):
        rc, tr = a * N, (1 - a) * N
        tc = (c0 + c1 * rc + c2 * rc * rc) if rc > 0 else 0.0
        tt = (t0 + t1 * tr) if tr > 0 else 0.0
        return max(tc, tt)

    valid_roots = []
    if abs(qa) < 1e-12:
        if abs(qb) > 1e-12:
            r = -qc / qb
            if 0.0 <= r <= 1.0:
                valid_roots.append(r)
    else:
        disc = qb * qb - 4 * qa * qc
        if disc >= 0:
            sd = math.sqrt(disc)
            for r in [(-qb + sd) / (2 * qa), (-qb - sd) / (2 * qa)]:
                if 0.0 <= r <= 1.0:
                    valid_roots.append(r)

    if valid_roots:
        alpha_cont = min(valid_roots, key=_t_total_at)
    else:
        alpha_cont = 0.0 if t_all_transfer < t_all_compute else 1.0

    # snap: evaluate both neighboring block boundaries, pick the better one
    num_blocks = math.ceil(N / block_size)
    sb_raw = alpha_cont * num_blocks
    candidates = set()
    candidates.add(max(0, min(math.floor(sb_raw), num_blocks)))
    candidates.add(max(0, min(math.ceil(sb_raw), num_blocks)))

    def eval_t_total(sb):
        rc = min(sb * block_size, N)
        tr = N - rc
        t_comp = (c0 + c1 * rc + c2 * rc * rc) if rc > 0 else 0.0
        t_tran = (t0 + t1 * tr) if tr > 0 else 0.0
        return max(t_comp, t_tran)

    best_sb = min(candidates, key=eval_t_total)
    return min(best_sb * block_size, N) / N if N > 0 else 0.0


def predict_all(Ns: list[int], block_size: int,
                c_params: tuple, t_params: tuple) -> dict[int, float]:
    """Return {N: α*(N)} for every N in the list."""
    return {N: predict_alpha_star(N, block_size, c_params, t_params) for N in Ns}


# ====================================================================== #
#  Experiment 2 (profile variant): per-N optimal α
# ====================================================================== #

def run_exp2_profile(engine: MigrationEngine, prompt_tokens: list[int],
                     alpha_map: dict[int, float], num_repeats: int) -> list[dict]:
    """Like run_exp2, but hybrid uses α*(N) from the fitted model."""
    Ns = sorted(alpha_map.keys())
    results = []

    for N in Ns:
        if N > len(prompt_tokens):
            if engine.rank == 1:
                print(f"  Skipping N={N}: prompt too short")
            continue

        token_ids = prompt_tokens[:N]
        strategies = {"kv_migration": 0.0, "token_migration": 1.0,
                      "hybrid": alpha_map[N]}

        for name, alpha in strategies.items():
            run_one(engine, token_ids, alpha)  # warmup
            times = [run_one(engine, token_ids, alpha) for _ in range(num_repeats)]

            if engine.rank == 1:
                times.sort()
                r = dict(N=N, strategy=name, alpha=alpha,
                         median=times[len(times) // 2],
                         min=min(times), max=max(times), all=times)
                results.append(r)
                print(f"  N={N} {name:20s} (α={alpha:.2f}): "
                      f"median={r['median']:.2f} ms  "
                      f"[{r['min']:.2f}, {r['max']:.2f}]")

        # brute-force optimal α for this N
        bf = bruteforce_optimal(engine, token_ids, num_repeats)
        if engine.rank == 1 and bf:
            results.append(bf)
            print(f"  N={N} {'hybrid_bruteforce':20s} (α={bf['alpha']:.2f}): "
                  f"median={bf['median']:.2f} ms  "
                  f"[{bf['min']:.2f}, {bf['max']:.2f}]")
    return results


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
                xytext=(alphas[min_idx], medians[min_idx] * 1.18),
                fontsize=9, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlabel("α (recompute fraction)", fontsize=12)
    ax.set_ylabel("T_total (ms)", fontsize=12)
    ax.set_title(f"T_total vs α  (N = {N})", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


def plot_exp2(results: list[dict], alpha_star, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(alpha_star, dict):
        hybrid_label = "Hybrid (α* per N)"
    else:
        hybrid_label = f"Hybrid (α={alpha_star:.2f})"
    labels = {"kv_migration": "KV Migration (α=0)",
              "token_migration": "Token Migration (α=1)",
              "hybrid": hybrid_label,
              "hybrid_bruteforce": "Hybrid (brute-force α*)"}
    colors = {"kv_migration": "#1f77b4", "token_migration": "#ff7f0e",
              "hybrid": "#2ca02c", "hybrid_bruteforce": "#d62728"}
    markers = {"kv_migration": "s", "token_migration": "^",
               "hybrid": "o", "hybrid_bruteforce": "D"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for strat in ["kv_migration", "token_migration", "hybrid", "hybrid_bruteforce"]:
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
#  Experiment 3: end-to-end with iterative catch-up
# ====================================================================== #

def run_one_iterative(engine: MigrationEngine, token_ids: list[int],
                      alpha: float, max_decode: int,
                      alpha_fn=None,
                      min_catchup_tokens: int = 1024) -> tuple[float, dict]:
    """Run one iterative migration. Returns (T_total_ms, stats) on rank 1."""
    src_seq = None
    if engine.rank == 0:
        src_seq = engine.prefill_src(token_ids)
    t, dst_seq, stats = engine.migrate_iterative(
        token_ids, alpha, src_seq=src_seq, max_decode=max_decode,
        alpha_fn=alpha_fn, min_catchup_tokens=min_catchup_tokens)
    dist.barrier()
    if engine.rank == 0:
        engine.cleanup_src(src_seq)
    else:
        engine.cleanup_dst(dst_seq)
    torch.cuda.empty_cache()
    return (t, stats) if engine.rank == 1 else (0.0, stats)


def verify_iterative_migration(engine: MigrationEngine, token_ids: list[int],
                                alpha: float, max_decode: int) -> bool:
    """方案B：独立比较 dst 生成的 max_decode 个 token 与纯 src 解码结果。

    rank 0: prefill + 独立解码 max_decode 个 token → 基线
    rank 1: 执行 migrate_iterative，dst 端得到 N + max_decode 个 token
    比较 dst 端新增的 max_decode 个 token 是否与基线一致。
    """
    N = len(token_ids)
    if engine.rank == 0:
        # 1. prefill 构建 KV cache
        src_seq = engine.prefill_src(token_ids)
        # 2. 独立解码 max_decode 个 token 作为基线
        baseline = engine.greedy_decode(engine.src, engine.src_bm, src_seq, max_decode)
        device = f"cuda:{engine.src.rank}"
        baseline_t = torch.tensor(baseline, dtype=torch.int64, device=device)
        # 发送基线给 rank 1
        dist.send(baseline_t, dst=1)
        engine.cleanup_src(src_seq)
        return True
    else:   # rank == 1
        # 1. 执行迭代迁移（内部会由 dst 解码完剩余的 max_decode - src_decoded 个 token）
        _, dst_seq, stats = engine.migrate_iterative(
            token_ids, alpha, src_seq=None, max_decode=max_decode)
        # 2. 取出迁移后 dst 新增的所有 token（N 之后的部分）
        migrated = dst_seq.token_ids[N:]   # 长度应为 max_decode
        # 3. 接收基线
        device = f"cuda:{engine.dst.rank}"
        baseline_t = torch.empty(max_decode, dtype=torch.int64, device=device)
        dist.recv(baseline_t, src=0)
        baseline = baseline_t.tolist()
        engine.cleanup_dst(dst_seq)
        ok = (migrated == baseline)
        if not ok:
            print(f"  [rank 1] ITERATIVE MISMATCH: "
                  f"baseline={baseline[:10]}... "
                  f"migrated={migrated[:10]}...")
        return ok


def run_exp3(engine: MigrationEngine, prompt_tokens: list[int],
             alpha_star: float, num_repeats: int,
             alpha_fn=None,
             min_catchup_tokens: int = 1024) -> list[dict]:
    """End-to-end benchmark: prefill N + decode N with iterative catch-up."""
    Ns = [1024, 2048, 4096, 8192, 8192 * 2]
    strategies = {"kv_migration": 0.0, "token_migration": 1.0, "hybrid": alpha_star}
    results = []

    for N in Ns:
        if N > len(prompt_tokens):
            if engine.rank == 1:
                print(f"  Skipping N={N}: prompt too short ({len(prompt_tokens)} tokens)")
            continue

        token_ids = prompt_tokens[:N]
        max_decode = N  # decode exactly N tokens

        for name, alpha in strategies.items():
            # alpha_fn only applies to hybrid strategy
            fn = alpha_fn if name == "hybrid" else None
            # warmup
            run_one_iterative(engine, token_ids, alpha, max_decode,
                              fn, min_catchup_tokens)

            times = []
            all_stats = []
            for _ in range(num_repeats):
                t, stats = run_one_iterative(engine, token_ids, alpha, max_decode,
                                             fn, min_catchup_tokens)
                times.append(t)
                all_stats.append(stats)

            if engine.rank == 1:
                times.sort()
                mid = len(times) // 2
                r = dict(N=N, strategy=name, alpha=alpha,
                         median=times[mid], min=min(times), max=max(times),
                         all=times,
                         num_rounds=all_stats[mid]["num_rounds"],
                         src_decoded=all_stats[mid]["src_decoded"],
                         dst_decoded=all_stats[mid]["dst_decoded"])
                results.append(r)
                print(f"  N={N} {name:20s} (α={alpha:.2f}): "
                      f"median={r['median']:.2f} ms  "
                      f"[{r['min']:.2f}, {r['max']:.2f}]  "
                      f"rounds={r['num_rounds']} "
                      f"src={r['src_decoded']} dst={r['dst_decoded']}")

    return results


def plot_exp3(results: list[dict], alpha_star: float, output_path: str):
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

    ax.set_xlabel("Sequence Length N (prefill = decode = N)", fontsize=12)
    ax.set_ylabel("T_total (ms)", fontsize=12)
    ax.set_title("End-to-End Migration Time (iterative catch-up)", fontsize=14)
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
        max_model_len=8192 * 2,
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
    engine = MigrationEngine(rank, src_runner, dst_runner, src_bm, dst_bm,
                             bandwidth_scale=args.bandwidth_scale)

    # -- tokenize a real prompt, ensure length == max_model_len --
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    chunk = "The quick brown fox jumps over the lazy dog. "
    chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
    prompt_tokens = (chunk_tokens * (config.max_model_len // len(chunk_tokens) + 2))[:config.max_model_len]
    assert len(prompt_tokens) == config.max_model_len, \
        f"got {len(prompt_tokens)}, need {config.max_model_len}"
    if rank == 1:
        print(f"Prompt tokens: {len(prompt_tokens)}")

    # ============================================================ #
    #  Correctness check: α=0 (pure transfer), 0.5 (hybrid), 1 (pure recompute)
    # ============================================================ #
    if rank == 1:
        print("\n=== Correctness check ===")
    N_check = min(1024, len(prompt_tokens))
    all_ok = True
    for check_alpha in [0.0, 0.5, 1.0]:
        ok = verify_migration(engine, prompt_tokens[:N_check], alpha=check_alpha)
        if rank == 1:
            label = {0.0: "pure transfer", 0.5: "hybrid", 1.0: "pure recompute"}[check_alpha]
            print(f"  α={check_alpha} ({label}): {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_ok = False
    ok_tensor = torch.tensor([int(all_ok)], dtype=torch.int32, device=f"cuda:{gpu_id}")
    dist.broadcast(ok_tensor, src=1)
    if ok_tensor.item() == 0:
        if rank == 1:
            print("  Aborting due to correctness failure.")
        dist.destroy_process_group()
        return

    # ============================================================ #
    #  Branch: --profile vs --no-profile
    # ============================================================ #
    exp2_Ns = [1024, 2048, 4096, 8192, 8192 * 2]

    if args.profile:
        # ====== PROFILE PATH ====== #

        # --- Step 1: dense profiling ---
        if rank == 1:
            print("\n=== Profiling (dense N sampling) ===")
        compute_data, transfer_data = run_profiling(
            engine, prompt_tokens, num_repeats=args.num_repeats)

        # --- Step 2: fit models (rank 1 only, broadcast params) ---
        c_params = (0.0, 0.0, 0.0)
        t_params = (0.0, 0.0)
        if rank == 1:
            c_params, t_params = fit_latency_models(compute_data, transfer_data)
            print(f"\n  Fitted T_compute(n) = {c_params[0]:.4f} + {c_params[1]:.6f}·n "
                  f"+ {c_params[2]:.10f}·n²")
            print(f"  Fitted T_transfer(n) = {t_params[0]:.4f} + {t_params[1]:.6f}·n")

        param_tensor = torch.zeros(5, dtype=torch.float64, device=f"cuda:{gpu_id}")
        if rank == 1:
            param_tensor[:3] = torch.tensor(c_params, dtype=torch.float64)
            param_tensor[3:] = torch.tensor(t_params, dtype=torch.float64)
        dist.broadcast(param_tensor, src=1)
        c_params = tuple(param_tensor[:3].tolist())
        t_params = tuple(param_tensor[3:].tolist())

        # --- Step 3: predict α*(N) for all target Ns ---
        block_size = engine.block_size
        alpha_map = predict_all(exp2_Ns, block_size, c_params, t_params)

        # choose exp1 N: where T_transfer ≈ T_compute (curves cross)
        if args.exp1_n:
            chosen_N = args.exp1_n
        else:
            # pick N from profiling data where ratio is closest to 1
            chosen_N = 4096
            if rank == 1 and compute_data and transfer_data:
                best_ratio = float("inf")
                for (nc, tc), (nt, tt) in zip(compute_data, transfer_data):
                    r = max(tc, tt) / max(min(tc, tt), 1e-6)
                    if r < best_ratio:
                        best_ratio = r
                        chosen_N = nc
            n_tensor = torch.tensor([chosen_N], dtype=torch.int64, device=f"cuda:{gpu_id}")
            dist.broadcast(n_tensor, src=1)
            chosen_N = n_tensor.item()

        predicted_alpha = predict_alpha_star(chosen_N, block_size, c_params, t_params)

        if rank == 1:
            print(f"\n  Predicted α* for exp1 N={chosen_N}: {predicted_alpha:.3f}")
            for N in exp2_Ns:
                if N in alpha_map:
                    print(f"  Predicted α*(N={N}): {alpha_map[N]:.3f}")

        # --- Step 4: exp1 — α sweep + compare with prediction ---
        if rank == 1:
            print(f"\n=== Experiment 1: α sweep at N={chosen_N} ===")
        exp1_tokens = prompt_tokens[:chosen_N]
        exp1_results = run_exp1(engine, exp1_tokens, args.num_repeats)

        alpha_star_bruteforce = 0.5
        if rank == 1 and exp1_results:
            best = min(exp1_results, key=lambda r: r["median"])
            alpha_star_bruteforce = best["alpha"]
            print(f"  Brute-force α* = {alpha_star_bruteforce:.1f}, "
                  f"Predicted α* = {predicted_alpha:.3f}")

        # --- Step 5: exp2 — N sweep with per-N α*(N) ---
        if rank == 1:
            print(f"\n=== Experiment 2: N sweep (per-N α*) ===")
        exp2_results = run_exp2_profile(engine, prompt_tokens, alpha_map, args.num_repeats)

        # --- Collect ---
        profile_info = {
            "c_params": list(c_params), "t_params": list(t_params),
            "alpha_map": {str(k): v for k, v in alpha_map.items()},
            "predicted_alpha_exp1": predicted_alpha,
            "bruteforce_alpha_exp1": alpha_star_bruteforce,
        }
        alpha_star_out = alpha_map  # dict for plot label

    else:
        # ====== NO-PROFILE PATH (original) ====== #
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

        if rank == 1:
            print(f"\n=== Experiment 1: α sweep at N={chosen_N} ===")
        exp1_tokens = prompt_tokens[:chosen_N]
        exp1_results = run_exp1(engine, exp1_tokens, args.num_repeats)

        alpha_star = 0.5
        if rank == 1 and exp1_results:
            best = min(exp1_results, key=lambda r: r["median"])
            alpha_star = best["alpha"]
            print(f"  → α* = {alpha_star:.1f}")

        a_tensor = torch.tensor([alpha_star], dtype=torch.float32, device=f"cuda:{gpu_id}")
        dist.broadcast(a_tensor, src=1)
        alpha_star = a_tensor.item()

        if rank == 1:
            print(f"\n=== Experiment 2: N sweep at α*={alpha_star:.2f} ===")
        exp2_results = run_exp2(engine, prompt_tokens, alpha_star, args.num_repeats)

        profile_info = None
        alpha_star_out = alpha_star

    # ============================================================ #
    #  Experiment 3: iterative catch-up (optional)
    # ============================================================ #
    exp3_results = []
    if args.iterative:
        # resolve α* for exp3
        if isinstance(alpha_star_out, dict):
            exp3_alpha = profile_info["bruteforce_alpha_exp1"]
        else:
            exp3_alpha = alpha_star_out

        # build per-round alpha_fn if profiling data available
        exp3_alpha_fn = None
        if profile_info:
            _c = tuple(profile_info["c_params"])
            _t = tuple(profile_info["t_params"])
            _bs = engine.block_size
            exp3_alpha_fn = lambda M, c=_c, t=_t, bs=_bs: predict_alpha_star(M, bs, c, t)

        min_ct = args.min_catchup_tokens

        # correctness check
        if rank == 1:
            print("\n=== Exp3 correctness check (iterative, α=0.5) ===")
        N_check = min(1024, len(prompt_tokens))
        ok = verify_iterative_migration(
            engine, prompt_tokens[:N_check], alpha=0.5, max_decode=N_check)
        if rank == 1:
            print(f"  Result: {'PASS' if ok else 'FAIL'}")

        if rank == 1:
            mode = "per-round α* from profiling" if exp3_alpha_fn else f"fixed α*={exp3_alpha:.2f}"
            print(f"\n=== Experiment 3: iterative catch-up ({mode}, "
                  f"min_catchup={min_ct}) ===")
        exp3_results = run_exp3(engine, prompt_tokens, exp3_alpha, args.num_repeats,
                                alpha_fn=exp3_alpha_fn, min_catchup_tokens=min_ct)

    # ============================================================ #
    #  Collect results
    # ============================================================ #
    if rank == 1:
        data = {
            "exp1": exp1_results,
            "exp2": exp2_results,
            "chosen_N": chosen_N,
            "alpha_star": alpha_star_out if not isinstance(alpha_star_out, dict)
                          else profile_info["bruteforce_alpha_exp1"],
        }
        if exp3_results:
            data["exp3"] = exp3_results
        if profile_info:
            data["profile"] = profile_info

        if result_queue is not None:
            result_queue.put(data)
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            json_path = os.path.join(args.output_dir, "results.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nRaw results → {json_path}")
            if data["exp1"]:
                plot_exp1(data["exp1"], chosen_N,
                          os.path.join(args.output_dir, "exp1_alpha_sweep.png"))
            if data["exp2"]:
                plot_exp2(data["exp2"], alpha_star_out,
                          os.path.join(args.output_dir, "exp2_n_sweep.png"))
            if data.get("exp3"):
                plot_exp3(data["exp3"], data["alpha_star"],
                          os.path.join(args.output_dir, "exp3_iterative.png"))

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
    parser.add_argument("--bandwidth-scale", type=int, default=1,
                        help="Inflate transfer buffer Kx to simulate 1/K bandwidth. "
                             "K=4 recommended for A100 PCIe.")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable profiling: fit T_compute/T_transfer models, "
                             "predict optimal α*(N) analytically.")
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    parser.add_argument("--iterative", action="store_true", default=False,
                        help="Run exp3: end-to-end migration with iterative catch-up.")
    parser.add_argument("--min-catchup-tokens", type=int, default=1024,
                        help="When remaining tokens < this, final round uses pure "
                             "KV transfer and exits catch-up (default 1024).")
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

        try:
            data = result_queue.get(timeout=60)
        except Exception:
            print("No results collected (queue timeout).")
            return

        json_path = os.path.join(args.output_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nRaw results → {json_path}")

        N = data["chosen_N"]
        if data["exp1"]:
            plot_exp1(data["exp1"], N, os.path.join(args.output_dir, "exp1_alpha_sweep.png"))
        if data["exp2"]:
            if "profile" in data and data["profile"]:
                alpha_map = {int(k): v for k, v in data["profile"]["alpha_map"].items()}
                plot_exp2(data["exp2"], alpha_map,
                          os.path.join(args.output_dir, "exp2_n_sweep.png"))
            else:
                plot_exp2(data["exp2"], data["alpha_star"],
                          os.path.join(args.output_dir, "exp2_n_sweep.png"))
        if data.get("exp3"):
            plot_exp3(data["exp3"], data["alpha_star"],
                      os.path.join(args.output_dir, "exp3_iterative.png"))


if __name__ == "__main__":
    main()
