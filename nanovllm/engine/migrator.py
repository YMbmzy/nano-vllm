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
        # NOTE: the last block may be partially filled — trailing slots carry
        # uninitialised memory from src. Functionally safe because dst's
        # subsequent decode overwrites those slots. If a future attention
        # kernel requires padding slots to be zeroed, zero the tail here.
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
                # ensure unpack copy_ on default stream completes before
                # NCCL recv overwrites buf on the NCCL stream
                torch.cuda.current_stream().synchronize()
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
    #  Decode one token on src (rank 0 only, used by iterative migration)
    # ------------------------------------------------------------------ #
    def decode_one_src(self, seq: Sequence) -> int:
        assert self.rank == 0
        self.src_bm.may_append(seq)
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        with torch.cuda.device(self.src.rank):
            input_ids, positions = self.src.prepare_decode([seq])
            logits = self.src.run_model(input_ids, positions, is_prefill=False)
            reset_context()
        token = logits[0].argmax().item()
        seq.num_cached_tokens += 1
        seq.append_token(token)
        return token

    # ------------------------------------------------------------------ #
    #  Iterative catch-up migration (exp3) — pure KV transfer per round
    # ------------------------------------------------------------------ #
    def migrate_iterative(self, token_ids: list[int],
                          src_seq: Sequence | None = None,
                          max_decode: int | None = None,
                          ) -> tuple[float, 'Sequence | None', dict]:
        """Pure KV migration with iterative catch-up.

        Each round:
          rank 0: pack & send KV for pending tokens, decode until dst round-done
          rank 1: recv KV, store into cache

        Exit: when gap (unsent tokens) <= 1, src stops decoding, transfers
        the final token's KV, migration complete. Then dst decodes remaining.

        Returns (T_total_ms, dst_seq, stats).
        """
        N = len(token_ids)
        if max_decode is None:
            max_decode = N
        K = self.bandwidth_scale

        all_tokens = list(token_ids)
        migrated_up_to = 0
        src_decoded = 0
        round_num = 0
        device = self.src.rank if self.rank == 0 else self.dst.rank

        # rank 1: allocate dst blocks for initial N tokens
        dst_seq = None
        if self.rank == 1:
            dst_seq = Sequence(token_ids)
            for _ in range(dst_seq.num_blocks):
                dst_seq.block_table.append(self.dst_bm._allocate_block())

        t_start = None

        while True:
            round_end = len(all_tokens)
            M = round_end - migrated_up_to
            if M == 0:
                break

            # src decodes only if budget remains AND gap > 1
            should_decode = (src_decoded < max_decode) and (M > 1)

            # block range (floor start to re-send partial block with new slots)
            start_block = migrated_up_to // self.block_size
            end_block = math.ceil(round_end / self.block_size)
            num_transfer_blocks = end_block - start_block

            # allocate buffer
            buf = None
            if num_transfer_blocks > 0:
                buf = torch.empty(
                    *self.kv_block_shape[:-3], num_transfer_blocks,
                    *self.kv_block_shape[-3:],
                    dtype=self.kv_dtype, device=f"cuda:{device}")

            # rank 0: ensure blocks exist on src, pack into buffer
            if self.rank == 0 and buf is not None:
                while len(src_seq.block_table) < end_block:
                    src_seq.block_table.append(self.src_bm._allocate_block())
                with torch.cuda.device(device):
                    for i in range(num_transfer_blocks):
                        src_bid = src_seq.block_table[start_block + i]
                        buf[:, :, i].copy_(self.src.kv_cache[:, :, src_bid])
                    torch.cuda.synchronize()

            dist.barrier()
            if t_start is None and self.rank == 1:
                t_start = time.perf_counter()

            new_tokens = []
            done_t = torch.empty(1, dtype=torch.int32, device=f"cuda:{device}")

            if self.rank == 0:
                if buf is not None:
                    for _ in range(K):
                        dist.send(buf, dst=1)

                # Round-done signal from dst: decode only until dst finishes
                # receiving/unpacking this round's KV.
                done_recv = dist.irecv(done_t, src=1)
                if should_decode:
                    while src_decoded < max_decode:
                        if done_recv.is_completed():
                            break
                        token = self.decode_one_src(src_seq)
                        new_tokens.append(token)
                        src_decoded += 1
                done_recv.wait()

            else:  # rank 1 (dst)
                with torch.cuda.device(device):
                    # first recv matches rank 0's blocking send
                    if buf is not None:
                        recv_work = dist.irecv(buf, src=0)
                        recv_work.wait()
                        for i in range(num_transfer_blocks):
                            dst_bid = dst_seq.block_table[start_block + i]
                            self.dst.kv_cache[:, :, dst_bid].copy_(buf[:, :, i])

                    # K-1 padding recvs (blocking, matches rank 0 blocking sends)
                    if buf is not None:
                        torch.cuda.current_stream().synchronize()
                        for _ in range(K - 1):
                            dist.recv(buf, src=0)

                    torch.cuda.synchronize()
                    done_t.fill_(1)
                    dist.send(done_t, dst=0)

            # exchange new tokens (blocking send/recv)
            if self.rank == 0:
                cnt_t = torch.tensor([len(new_tokens)], dtype=torch.int64, device=f"cuda:{device}")
                dist.send(cnt_t, dst=1)
            else:
                cnt_t = torch.empty(1, dtype=torch.int64, device=f"cuda:{device}")
                dist.recv(cnt_t, src=0)
            new_count = cnt_t.item()

            if new_count > 0:
                if self.rank == 0:
                    ids_t = torch.tensor(new_tokens, dtype=torch.int64, device=f"cuda:{device}")
                    dist.send(ids_t, dst=1)
                else:
                    ids_t = torch.empty(new_count, dtype=torch.int64, device=f"cuda:{device}")
                    dist.recv(ids_t, src=0)
                new_ids = ids_t.tolist() if self.rank == 1 else new_tokens
                all_tokens.extend(new_ids)

                if self.rank == 1:
                    for tid in new_ids:
                        dst_seq.token_ids.append(tid)
                        dst_seq.num_tokens += 1
                        dst_seq.last_token = tid
                    while len(dst_seq.block_table) < dst_seq.num_blocks:
                        dst_seq.block_table.append(self.dst_bm._allocate_block())

            migrated_up_to = round_end
            round_num += 1
            if self.rank == 1:
                src_decoded += new_count

        # ---- Phase 2: dst decodes remaining tokens ----
        dst_decoded = 0
        if self.rank == 1:
            dst_seq.num_cached_tokens = len(all_tokens)
            dst_seq.num_scheduled_tokens = 0
            dst_seq.is_prefill = False

            remaining = max_decode - src_decoded
            if remaining > 0:
                with torch.cuda.device(device):
                    for _ in range(remaining):
                        self.dst_bm.may_append(dst_seq)
                        dst_seq.num_scheduled_tokens = 1
                        dst_seq.is_prefill = False
                        inp, pos = self.dst.prepare_decode([dst_seq])
                        logits = self.dst.run_model(inp, pos, is_prefill=False)
                        reset_context()
                        token = logits[0].argmax().item()
                        dst_seq.num_cached_tokens += 1
                        dst_seq.append_token(token)
                        dst_decoded += 1
                    torch.cuda.synchronize()

        if self.rank == 1 and t_start is not None:
            t_total = (time.perf_counter() - t_start) * 1000
        else:
            t_total = 0.0

        stats = {"num_rounds": round_num, "src_decoded": src_decoded,
                 "dst_decoded": dst_decoded}
        return t_total, dst_seq, stats
    
    # ------------------------------------------------------------------ #
    #  Baseline 1 — Token Migration (src idles; dst re-prefills + decodes)
    # ------------------------------------------------------------------ #
    def migrate_token_migration(self, token_ids: list[int],
                                src_seq: Sequence | None = None,
                                max_decode: int | None = None,
                                ) -> tuple[float, 'Sequence | None', dict]:
        """Baseline: src ships token IDs, dst rebuilds KV via prefill and decodes.

        Semantics: migration is triggered right after src finishes prefill.
        src does no useful work during migration; dst handles all max_decode steps.
        T_migration = T_send_tokens(~0) + T_prefill(N) on dst + T_decode(max_decode) on dst.
        """
        N = len(token_ids)
        if max_decode is None:
            max_decode = N
        device = self.src.rank if self.rank == 0 else self.dst.rank

        # ---- sync: migration trigger ----
        dist.barrier()
        t_start = time.perf_counter() if self.rank == 1 else None

        # Phase 1: src sends token IDs to dst (src idles otherwise)
        if self.rank == 0:
            total_t = torch.tensor([N], dtype=torch.int64, device=f"cuda:{device}")
            dist.send(total_t, dst=1)
            ids_t = torch.tensor(token_ids, dtype=torch.int64, device=f"cuda:{device}")
            dist.send(ids_t, dst=1)
        else:
            total_t = torch.empty(1, dtype=torch.int64, device=f"cuda:{device}")
            dist.recv(total_t, src=0)
            ids_t = torch.empty(total_t.item(), dtype=torch.int64, device=f"cuda:{device}")
            dist.recv(ids_t, src=0)

        # Phase 2: dst re-prefills the N tokens
        dst_seq = None
        if self.rank == 1:
            recv_ids = ids_t.tolist()
            with torch.cuda.device(device):
                dst_seq = Sequence(recv_ids)
                for _ in range(dst_seq.num_blocks):
                    dst_seq.block_table.append(self.dst_bm._allocate_block())
                dst_seq.num_cached_tokens = 0
                dst_seq.num_scheduled_tokens = N
                inp, pos = self.dst.prepare_prefill([dst_seq])
                self.dst.run_model(inp, pos, is_prefill=True)
                reset_context()
                dst_seq.num_cached_tokens = N
                dst_seq.num_scheduled_tokens = 0
                dst_seq.is_prefill = False

        # Phase 3: dst decodes max_decode steps
        dst_decoded = 0
        if self.rank == 1:
            with torch.cuda.device(device):
                for _ in range(max_decode):
                    self.dst_bm.may_append(dst_seq)
                    dst_seq.num_scheduled_tokens = 1
                    dst_seq.is_prefill = False
                    inp, pos = self.dst.prepare_decode([dst_seq])
                    logits = self.dst.run_model(inp, pos, is_prefill=False)
                    reset_context()
                    token = logits[0].argmax().item()
                    dst_seq.num_cached_tokens += 1
                    dst_seq.append_token(token)
                    dst_decoded += 1
                torch.cuda.synchronize()

        t_total = (time.perf_counter() - t_start) * 1000 if self.rank == 1 else 0.0
        stats = {"num_rounds": 0, "src_decoded": 0, "dst_decoded": dst_decoded}
        return t_total, dst_seq, stats

    # ------------------------------------------------------------------ #
    #  Baseline 2 — Deferred KV (src idles; dst receives KV then decodes)
    # ------------------------------------------------------------------ #
    def migrate_deferred_kv(self, token_ids: list[int],
                            src_seq: Sequence | None = None,
                            max_decode: int | None = None,
                            ) -> tuple[float, 'Sequence | None', dict]:
        """Baseline: src ships complete KV (no overlap); dst decodes.

        Semantics: migration is triggered right after src finishes prefill.
        src does no decode during migration — only ships the KV it already has.
        T_migration = T_transfer(N) + T_decode(max_decode) on dst.
        """
        N = len(token_ids)
        if max_decode is None:
            max_decode = N
        K = self.bandwidth_scale
        device = self.src.rank if self.rank == 0 else self.dst.rank

        num_blocks = math.ceil(N / self.block_size)

        # Pre-pack src KV BEFORE the timer starts is wrong — packing happens
        # during migration. But we do allocate the buffer outside to match
        # the iterative path's accounting.
        buf = torch.empty(
            *self.kv_block_shape[:-3], num_blocks,
            *self.kv_block_shape[-3:],
            dtype=self.kv_dtype, device=f"cuda:{device}")

        # ---- sync: migration trigger ----
        dist.barrier()
        t_start = time.perf_counter() if self.rank == 1 else None

        # Phase 1: send token IDs (so dst can build Sequence)
        if self.rank == 0:
            total_t = torch.tensor([N], dtype=torch.int64, device=f"cuda:{device}")
            dist.send(total_t, dst=1)
            ids_t = torch.tensor(token_ids, dtype=torch.int64, device=f"cuda:{device}")
            dist.send(ids_t, dst=1)
        else:
            total_t = torch.empty(1, dtype=torch.int64, device=f"cuda:{device}")
            dist.recv(total_t, src=0)
            ids_t = torch.empty(total_t.item(), dtype=torch.int64, device=f"cuda:{device}")
            dist.recv(ids_t, src=0)

        # Phase 2: src packs KV and transfers; dst receives and unpacks
        if self.rank == 0:
            with torch.cuda.device(device):
                for i in range(num_blocks):
                    src_bid = src_seq.block_table[i]
                    buf[:, :, i].copy_(self.src.kv_cache[:, :, src_bid])
                torch.cuda.synchronize()
            for _ in range(K):
                dist.send(buf, dst=1)

        dst_seq = None
        if self.rank == 1:
            recv_ids = ids_t.tolist()
            with torch.cuda.device(device):
                # recv real data (round 1)
                dist.recv(buf, src=0)
                # allocate dst blocks and unpack
                dst_seq = Sequence(recv_ids)
                for _ in range(dst_seq.num_blocks):
                    dst_seq.block_table.append(self.dst_bm._allocate_block())
                for i in range(num_blocks):
                    dst_bid = dst_seq.block_table[i]
                    self.dst.kv_cache[:, :, dst_bid].copy_(buf[:, :, i])
                # K-1 padding rounds
                torch.cuda.current_stream().synchronize()
                for _ in range(K - 1):
                    dist.recv(buf, src=0)
                torch.cuda.synchronize()
                dst_seq.num_cached_tokens = N
                dst_seq.num_scheduled_tokens = 0
                dst_seq.is_prefill = False

        # Phase 3: dst decodes max_decode steps
        dst_decoded = 0
        if self.rank == 1:
            with torch.cuda.device(device):
                for _ in range(max_decode):
                    self.dst_bm.may_append(dst_seq)
                    dst_seq.num_scheduled_tokens = 1
                    dst_seq.is_prefill = False
                    inp, pos = self.dst.prepare_decode([dst_seq])
                    logits = self.dst.run_model(inp, pos, is_prefill=False)
                    reset_context()
                    token = logits[0].argmax().item()
                    dst_seq.num_cached_tokens += 1
                    dst_seq.append_token(token)
                    dst_decoded += 1
                torch.cuda.synchronize()

        t_total = (time.perf_counter() - t_start) * 1000 if self.rank == 1 else 0.0
        stats = {"num_rounds": 0, "src_decoded": 0, "dst_decoded": dst_decoded}
        return t_total, dst_seq, stats
       
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
