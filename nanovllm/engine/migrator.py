import time
import math
import threading
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
    #  Iterative catch-up migration (exp3)
    # ------------------------------------------------------------------ #
    def migrate_iterative(self, token_ids: list[int], alpha: float,
                          src_seq: Sequence | None = None,
                          max_decode: int | None = None,
                          alpha_fn: 'Callable[[int], float] | None' = None,
                          min_catchup_tokens: int = 1024,
                          ) -> tuple[float, 'Sequence | None', dict]:
        """Full end-to-end migration with iterative catch-up.

        Phase 1: catch-up rounds — src sends KV + decodes, dst recvs + recomputes.
                 Exits when M < min_catchup_tokens (final round uses α=0 pure transfer).
        Phase 2: dst recomputes leftover tokens from last round.
        Phase 3: dst decodes remaining tokens (max_decode - src_decoded).

        Args:
            token_ids: initial prefilled token IDs (length N)
            alpha: hybrid split ratio (fallback when alpha_fn is None)
            src_seq: prefilled Sequence on rank 0
            max_decode: total decode tokens for src (default = len(token_ids))
            alpha_fn: optional callable(M) → α, for per-round optimal α from profiling
            min_catchup_tokens: when M < this, do pure KV transfer and exit (default 1024)

        Returns:
            (T_total_ms, dst_seq, stats_dict)
            - rank 0: (0.0, None, stats)
            - rank 1: (T_total, dst_seq with full 2N KV, stats)
        """
        N = len(token_ids)
        if max_decode is None:
            max_decode = N
        K = self.bandwidth_scale

        all_tokens = list(token_ids)   # grows as src decodes
        migrated_up_to = 0             # tokens whose KV dst already has
        src_decoded = 0                # total tokens src has decoded
        round_num = 0
        device = self.src.rank if self.rank == 0 else self.dst.rank

        # ---- rank 1: create dst_seq with blocks for initial N tokens ----
        dst_seq = None
        if self.rank == 1:
            dst_seq = Sequence(token_ids)
            for _ in range(dst_seq.num_blocks):
                dst_seq.block_table.append(self.dst_bm._allocate_block())

        # ============================================================ #
        #  Phase 1: iterative catch-up rounds
        #  Exit when M < min_catchup_tokens (that round uses α=0).
        #
        #  CAVEAT: for α=0 (pure KV) and α=1 (pure token), the catch-up
        #  protocol adds overhead without compute/transfer overlap.
        #  Also, the min_catchup_tokens exit forces α=0 on the final
        #  round regardless of strategy, so α=1 is not truly pure here.
        # ============================================================ #
        t_start = None  # set after first barrier

        while True:
            round_end = len(all_tokens)
            M = round_end - migrated_up_to
            if M == 0:
                break

            # -- resolve α for this round --
            force_exit = False
            if M < min_catchup_tokens and round_num > 0:
                round_alpha = 0.0    # pure transfer, skip compute overhead
                force_exit = True    # last catch-up round
            elif alpha_fn is not None:
                round_alpha = alpha_fn(M)
            else:
                round_alpha = alpha

            # -- compute split for this round's M tokens --
            split_idx = self.compute_split(M, round_alpha)

            # All block arithmetic in absolute coordinates to handle
            # non-block-aligned migrated_up_to correctly.
            end_token = migrated_up_to + M
            end_block_abs = math.ceil(end_token / self.block_size)

            # Snap split to absolute block boundary to avoid recompute/transfer
            # writing to the same block in parallel (race condition).
            abs_split_token = migrated_up_to + split_idx
            abs_transfer_start = math.ceil(abs_split_token / self.block_size)
            split_idx = min(abs_transfer_start * self.block_size - migrated_up_to, M)

            num_transfer_blocks = end_block_abs - abs_transfer_start
            abs_start_block = abs_transfer_start

            # -- prepare transfer buffer --
            # NOTE: last block's trailing slots may be uninitialised (see migrate()).
            if num_transfer_blocks > 0:
                buf = torch.empty(
                    *self.kv_block_shape[:-3], num_transfer_blocks,
                    *self.kv_block_shape[-3:],
                    dtype=self.kv_dtype, device=f"cuda:{device}")
            else:
                buf = None

            # -- rank 0: pack src KV --
            if self.rank == 0 and buf is not None:
                # src may have decoded new tokens in previous rounds.
                # Ensure block_table keeps up with current sequence length
                # before indexing absolute transfer blocks.
                while len(src_seq.block_table) < src_seq.num_blocks:
                    src_seq.block_table.append(self.src_bm._allocate_block())
                need_blocks = abs_start_block + num_transfer_blocks
                if need_blocks > len(src_seq.block_table):
                    raise RuntimeError(
                        "src block table too short for transfer: "
                        f"need={need_blocks}, have={len(src_seq.block_table)}, "
                        f"round={round_num}, M={M}, migrated_up_to={migrated_up_to}, "
                        f"src_tokens={src_seq.num_tokens}, src_blocks={src_seq.num_blocks}"
                    )
                with torch.cuda.device(device):
                    for i in range(num_transfer_blocks):
                        src_bid = src_seq.block_table[abs_start_block + i]
                        buf[:, :, i].copy_(self.src.kv_cache[:, :, src_bid])
                    torch.cuda.synchronize()

            # -- done signal buffer (rank 0 receives, rank 1 sends) --
            done_buf = torch.empty(1, dtype=torch.int32, device=f"cuda:{device}")

            # ---- barrier: start round ----
            dist.barrier()

            # start timer after first barrier — no cuda.synchronize here so
            # we don't exclude any GPU work that starts immediately after barrier
            if t_start is None:
                if self.rank == 1:
                    t_start = time.perf_counter()

            new_tokens = []

            if self.rank == 0:
                # --- src: blocking NCCL in thread + decode in main thread ---
                # NCCL is_completed() is unreliable for P2P ops; use threading
                # instead. Blocking NCCL calls release the GIL, so the main
                # thread can decode (Python + CUDA kernels) in parallel.
                src_device = self.src.rank
                def _nccl_src_work():
                    torch.cuda.set_device(src_device)
                    send_works = []
                    if buf is not None:
                        for _ in range(K):
                            send_works.append(dist.isend(buf, dst=1))
                    done_work = dist.irecv(done_buf, src=1)
                    for w in send_works:
                        w.wait()
                    done_work.wait()  # wait until dst signals this round done

                nccl_thread = threading.Thread(target=_nccl_src_work)
                nccl_thread.start()

                while nccl_thread.is_alive():
                    if src_decoded < max_decode:
                        token = self.decode_one_src(src_seq)
                        new_tokens.append(token)
                        src_decoded += 1
                    else:
                        nccl_thread.join()
                        break

                nccl_thread.join()

            else:  # rank == 1
                # --- dst: recv + recompute + unpack + signal done ---
                with torch.cuda.device(device):
                    recv_work = None
                    if buf is not None:
                        recv_work = dist.irecv(buf, src=0)

                    # recompute on compute stream
                    compute_stream = torch.cuda.Stream()
                    if split_idx > 0:
                        recompute_end = migrated_up_to + split_idx
                        recompute_blocks_end = abs_transfer_start  # = abs_start_block
                        # token_ids must be the full prefix [0, recompute_end)
                        # so prepare_prefill can index seq[num_cached_tokens:]
                        # and positions/RoPE start from the correct offset
                        recompute_seq = Sequence(all_tokens[:recompute_end])
                        recompute_seq.block_table = list(
                            dst_seq.block_table[:recompute_blocks_end])
                        recompute_seq.num_cached_tokens = migrated_up_to
                        recompute_seq.num_scheduled_tokens = split_idx
                        with torch.cuda.stream(compute_stream):
                            inp, pos = self.dst.prepare_prefill([recompute_seq])
                            self.dst.run_model(inp, pos, is_prefill=True)
                            reset_context()

                    # wait recv, unpack
                    if recv_work is not None:
                        recv_work.wait()
                        for i in range(num_transfer_blocks):
                            dst_bid = dst_seq.block_table[abs_start_block + i]
                            self.dst.kv_cache[:, :, dst_bid].copy_(buf[:, :, i])

                    # K-1 padding recvs
                    if buf is not None:
                        torch.cuda.current_stream().synchronize()
                        for _ in range(K - 1):
                            pad_work = dist.irecv(buf, src=0)
                            pad_work.wait()

                    compute_stream.synchronize()
                    torch.cuda.synchronize()

                    # signal src that round is done
                    done_buf.fill_(1)
                    done_send_work = dist.isend(done_buf, dst=0)
                    done_send_work.wait()

            # ---- exchange new tokens (pure P2P, avoid collective mixing) ----
            if self.rank == 0:
                new_count_t = torch.tensor(
                    [len(new_tokens)], dtype=torch.int64, device=f"cuda:{device}")
                count_send_work = dist.isend(new_count_t, dst=1)
                count_send_work.wait()
            else:
                new_count_t = torch.empty(1, dtype=torch.int64, device=f"cuda:{device}")
                dist.recv(new_count_t, src=0)
            new_count = new_count_t.item()

            if new_count > 0:
                if self.rank == 0:
                    new_ids_t = torch.tensor(
                        new_tokens, dtype=torch.int64, device=f"cuda:{device}")
                    ids_send_work = dist.isend(new_ids_t, dst=1)
                    ids_send_work.wait()
                else:
                    new_ids_t = torch.empty(
                        new_count, dtype=torch.int64, device=f"cuda:{device}")
                    dist.recv(new_ids_t, src=0)

                new_ids = new_ids_t.tolist()
                all_tokens.extend(new_ids)

                # rank 1: grow dst_seq
                if self.rank == 1:
                    for tid in new_ids:
                        dst_seq.token_ids.append(tid)
                        dst_seq.num_tokens += 1
                        dst_seq.last_token = tid
                    # allocate new blocks as needed
                    while len(dst_seq.block_table) < dst_seq.num_blocks:
                        dst_seq.block_table.append(self.dst_bm._allocate_block())

            migrated_up_to = round_end
            round_num += 1

            # also track src_decoded on rank 1 (from broadcast)
            if self.rank == 1:
                src_decoded += new_count

            if force_exit:
                break

        # ============================================================ #
        #  Phase 1.5: handoff — recompute leftover tokens on dst
        #  (tokens src decoded after last migrated round_end)
        # ============================================================ #
        leftover = len(all_tokens) - migrated_up_to
        if self.rank == 1 and leftover > 0:
            with torch.cuda.device(device):
                # allocate blocks for leftover tokens
                while len(dst_seq.block_table) < dst_seq.num_blocks:
                    dst_seq.block_table.append(self.dst_bm._allocate_block())
                # prefix-cached prefill for leftover tokens
                handoff_seq = Sequence(all_tokens[:migrated_up_to + leftover])
                handoff_seq.block_table = list(dst_seq.block_table)
                handoff_seq.num_cached_tokens = migrated_up_to
                handoff_seq.num_scheduled_tokens = leftover
                inp, pos = self.dst.prepare_prefill([handoff_seq])
                self.dst.run_model(inp, pos, is_prefill=True)
                reset_context()
        migrated_up_to = len(all_tokens)

        # ============================================================ #
        #  Phase 2: dst decodes remaining tokens
        # ============================================================ #
        dst_decoded = 0
        if self.rank == 1:
            # finalize dst_seq cached state
            dst_seq.num_cached_tokens = len(all_tokens)
            dst_seq.num_scheduled_tokens = 0
            dst_seq.is_prefill = False

            remaining = max_decode - src_decoded
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
