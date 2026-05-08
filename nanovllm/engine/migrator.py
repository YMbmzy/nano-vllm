import time
import math
import torch
import torch.distributed as dist

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


class MigrationEngine:
    """Orchestrates hybrid KV cache migration between two ModelRunner instances.

    Runs on BOTH ranks. Each rank calls the same methods; behavior differs by self.rank.
    """

    def __init__(self, rank: int, src_runner: ModelRunner | None, dst_runner: ModelRunner | None,
                 src_bm: BlockManager | None, dst_bm: BlockManager | None,
                 bandwidth_scale: int = 1):
        self.rank = rank
        self.src = src_runner      # only meaningful on rank 0
        self.dst = dst_runner      # only meaningful on rank 1
        self.src_bm = src_bm
        self.dst_bm = dst_bm
        self.bandwidth_scale = bandwidth_scale

        runner = src_runner if rank == 0 else dst_runner
        self.block_size = runner.block_size
        self.num_layers = runner.config.hf_config.num_hidden_layers
        self.kv_dtype = runner.kv_cache.dtype
        self.kv_block_shape = runner.kv_cache[:, :, 0].shape  # [2, num_layers, block_size, heads, dim]

    # ------------------------------------------------------------------ #
    #  α → split_idx (block-aligned)
    # ------------------------------------------------------------------ #
    def compute_split(self, N: int, alpha: float) -> int:
        num_blocks = math.ceil(N / self.block_size)
        split_block = round(alpha * num_blocks)
        split_block = max(0, min(split_block, num_blocks))
        return min(split_block * self.block_size, N)

    # ------------------------------------------------------------------ #
    #  Prefill on src to build KV cache (rank 0 only)
    # ------------------------------------------------------------------ #
    def prefill_src(self, token_ids: list[int]) -> Sequence:
        assert self.rank == 0
        seq = Sequence(token_ids)
        for _ in range(seq.num_blocks):
            seq.block_table.append(self.src_bm._allocate_block())
        seq.num_scheduled_tokens = len(token_ids)
        seq.num_cached_tokens = 0
        with torch.cuda.device(self.src.rank):
            input_ids, positions = self.src.prepare_prefill([seq])
            self.src.run_model(input_ids, positions, is_prefill=True)
            reset_context()
        seq.num_cached_tokens = len(token_ids)
        seq.num_scheduled_tokens = 0
        return seq

    # ------------------------------------------------------------------ #
    #  Core migration: called on BOTH ranks with the same (alpha, token_ids)
    # ------------------------------------------------------------------ #
    def migrate(self, token_ids: list[int], alpha: float,
                src_seq: Sequence | None = None) -> tuple[float, Sequence | None]:
        """Run one migration. Returns (T_total_ms, dst_seq).

        - rank 0: src_seq must be provided. Returns (0.0, None).
        - rank 1: src_seq is ignored. Returns (T_total, dst_seq).
        """
        N = len(token_ids)
        split_idx = self.compute_split(N, alpha)
        num_blocks = math.ceil(N / self.block_size)
        start_block = split_idx // self.block_size  # first block to transfer
        num_transfer_blocks = num_blocks - start_block

        # ---- rank 1: allocate dst blocks ----
        dst_seq = None
        if self.rank == 1:
            dst_seq = Sequence(token_ids)
            for _ in range(dst_seq.num_blocks):
                dst_seq.block_table.append(self.dst_bm._allocate_block())

        # ---- prepare contiguous transfer buffer on both ranks ----
        if num_transfer_blocks > 0:
            device = self.src.rank if self.rank == 0 else self.dst.rank
            buf = torch.empty(*self.kv_block_shape[:-3],   # [2, num_layers]
                              num_transfer_blocks,
                              *self.kv_block_shape[-3:],    # [block_size, heads, dim]
                              dtype=self.kv_dtype, device=f"cuda:{device}")
        else:
            buf = None

        # ---- rank 0: pack src KV into buffer ----
        if self.rank == 0 and buf is not None:
            with torch.cuda.device(self.src.rank):
                for i in range(num_transfer_blocks):
                    src_bid = src_seq.block_table[start_block + i]
                    buf[:, :, i].copy_(self.src.kv_cache[:, :, src_bid])
                torch.cuda.synchronize()

        K = self.bandwidth_scale

        # ==================== TIMED SECTION (rank 1) ==================== #
        dist.barrier()

        if self.rank == 0:
            if buf is not None:
                for _ in range(K):
                    dist.send(buf, dst=1)
            return 0.0, None

        # ---- rank 1: timed migration ----
        with torch.cuda.device(self.dst.rank):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            # -- async recv first round (real data) --
            recv_work = None
            if buf is not None:
                recv_work = dist.irecv(buf, src=0)

            # -- launch recompute (overlaps with all recv rounds) --
            compute_stream = torch.cuda.Stream()
            if split_idx > 0:
                recompute_seq = Sequence(token_ids[:split_idx])
                recompute_seq.block_table = list(dst_seq.block_table[:start_block])
                recompute_seq.num_cached_tokens = 0
                recompute_seq.num_scheduled_tokens = split_idx
                with torch.cuda.stream(compute_stream):
                    input_ids, positions = self.dst.prepare_prefill([recompute_seq])
                    self.dst.run_model(input_ids, positions, is_prefill=True)
                    reset_context()

            # -- wait first recv, unpack real data --
            if recv_work is not None:
                recv_work.wait()
                for i in range(num_transfer_blocks):
                    dst_bid = dst_seq.block_table[start_block + i]
                    self.dst.kv_cache[:, :, dst_bid].copy_(buf[:, :, i])

            # -- K-1 padding rounds: recv into same buf (data discarded) --
            if buf is not None:
                for _ in range(K - 1):
                    dist.recv(buf, src=0)

            # -- wait for recompute --
            compute_stream.synchronize()
            torch.cuda.synchronize()

            t_total = (time.perf_counter() - t_start) * 1000  # ms

        # finalize dst_seq state
        dst_seq.num_cached_tokens = N
        dst_seq.num_scheduled_tokens = 0
        dst_seq.is_prefill = False
        return t_total, dst_seq

    # ------------------------------------------------------------------ #
    #  Greedy decode (used for correctness check, runs on one rank)
    # ------------------------------------------------------------------ #
    def greedy_decode(self, runner: ModelRunner, bm: BlockManager,
                      seq: Sequence, num_tokens: int = 20) -> list[int]:
        device = runner.rank
        tokens = []
        with torch.cuda.device(device):
            for _ in range(num_tokens):
                bm.may_append(seq)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                input_ids, positions = runner.prepare_decode([seq])
                logits = runner.run_model(input_ids, positions, is_prefill=False)
                reset_context()
                token = logits[0].argmax().item()
                seq.num_cached_tokens += 1
                seq.append_token(token)
                tokens.append(token)
        return tokens

    # ------------------------------------------------------------------ #
    #  Cleanup dst blocks between runs
    # ------------------------------------------------------------------ #
    def cleanup_src(self, seq: Sequence):
        if seq is not None and self.src_bm is not None:
            self.src_bm.deallocate(seq)

    def cleanup_dst(self, seq: Sequence):
        if seq is not None and self.dst_bm is not None:
            self.dst_bm.deallocate(seq)
