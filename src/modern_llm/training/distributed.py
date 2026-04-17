"""Distributed training helpers.

Centralizes everything DDP-aware so train_lm/train_sft/train_dpo/train_verifier
do not have to duplicate boilerplate. Single-GPU mode is preserved: when
WORLD_SIZE is unset or equals 1, init_distributed() is a no-op and is_main()
always returns True.

Design choices:
- We treat env vars (WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
  as the source of truth. When run via torchrun these are already set.
- NCCL backend with bf16 communication hook on PCIe (no NVLink) consumer
  cards (RTX 4090 etc.) — set NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1 unless
  the user overrides them in their launcher.
- Per-rank seeding: seed = base_seed + rank to break IID dataloader sampling.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist


_INITIALIZED = False


def world_size() -> int:
    """Return WORLD_SIZE from env (1 if unset)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    """Return global RANK from env (0 if unset)."""
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    """Return LOCAL_RANK from env (0 if unset)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    """True if multi-process distributed training is in effect."""
    return world_size() > 1


def is_main_process() -> bool:
    """True on global rank 0 (or when not distributed)."""
    return rank() == 0


def _set_default_nccl_env() -> None:
    """Apply safe NCCL defaults for consumer PCIe boxes.

    - NCCL_P2P_DISABLE=1: RTX 4090 / 3090 do not support P2P over PCIe;
      leaving it on causes silent hangs.
    - NCCL_IB_DISABLE=1: most consumer boxes have no InfiniBand.
    - NCCL_ASYNC_ERROR_HANDLING=1: turn NCCL hangs into actionable
      Python exceptions instead of indefinite waits.
    Users can override any of these by setting them before launch.
    """
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


def init_distributed(backend: str = "nccl") -> None:
    """Initialize torch.distributed if WORLD_SIZE > 1.

    Pre: env vars RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
         are set (torchrun does this).
    Post: `dist.is_initialized()` is True; CUDA device is set to LOCAL_RANK.
    Idempotent: safe to call multiple times.
    """
    global _INITIALIZED
    if _INITIALIZED or not is_distributed():
        # Even in single-process CUDA mode, set device to LOCAL_RANK if available.
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
        _INITIALIZED = True
        return

    _set_default_nccl_env()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    _INITIALIZED = True


def cleanup_distributed() -> None:
    """Tear down the process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(base_seed: int) -> None:
    """Seed python/numpy/torch with `base_seed + rank` so each rank sees
    a different stream — important for dataloaders, dropout, sampling.
    """
    seed = base_seed + rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the device this rank should use."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank()}")
    return torch.device("cpu")


def wrap_ddp(
    model: torch.nn.Module,
    static_graph: bool = True,
    gradient_as_bucket_view: bool = True,
    bf16_comm_hook: bool = True,
) -> torch.nn.Module:
    """Wrap a model in DistributedDataParallel.

    Pre: init_distributed() has been called and model is on the correct device.
    Post: returns a DDP-wrapped model when distributed; otherwise returns
          the bare model unchanged.

    DDP flags:
        - gradient_as_bucket_view=True: zero-copy bucket view of grads,
          ~5–10% memory savings.
        - static_graph=True: enables additional comm/compute overlap by
          assuming the autograd graph is unchanged across iterations
          (true for our pure causal LM forward).
        - bf16 comm hook: bandwidth-halving allreduce on PCIe boxes,
          ~20–30% throughput lift on 2x4090.
    """
    if not is_distributed():
        return model

    from torch.nn.parallel import DistributedDataParallel as DDP

    ddp_model = DDP(
        model,
        device_ids=[local_rank()] if torch.cuda.is_available() else None,
        output_device=local_rank() if torch.cuda.is_available() else None,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )

    if bf16_comm_hook and torch.cuda.is_available():
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks

            ddp_model.register_comm_hook(state=None, hook=ddp_hooks.bf16_compress_hook)
        except Exception:
            # Older torch — silently skip the comm hook.
            pass

    return ddp_model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP and torch.compile wrappers to get the original module.

    Use this whenever saving a checkpoint or accessing config attributes,
    so consumers never see a `module.` or `_orig_mod.` prefix in keys.
    """
    inner = getattr(model, "module", model)  # DDP
    inner = getattr(inner, "_orig_mod", inner)  # torch.compile
    return inner


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor by mean across ranks. No-op when not distributed."""
    if not is_distributed() or not dist.is_initialized():
        return value
    t = value.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world_size()
    return t


@contextmanager
def main_process_first():
    """Context manager that runs the body on rank 0 first, then other ranks.

    Useful for one-time downloads / cache builds that should not race.
    """
    if is_distributed() and dist.is_initialized():
        if not is_main_process():
            dist.barrier()
        try:
            yield
        finally:
            if is_main_process():
                dist.barrier()
    else:
        yield


def barrier() -> None:
    """A no-op barrier when not distributed."""
    if is_distributed() and dist.is_initialized():
        dist.barrier()


def scale_grad_accum_for_world_size(
    desired_global_batch: int,
    micro_batch_size: int,
) -> int:
    """Compute gradient_accumulation_steps so the effective global batch size
    (in samples per optimizer step, summed across ranks) equals
    `desired_global_batch`.

    Math:
        global_batch = micro_batch * grad_accum * world_size
        => grad_accum = global_batch / (micro_batch * world_size)

    Pre: desired_global_batch is divisible by (micro_batch_size * world_size).
    Post: returns positive int >= 1.
    """
    denom = micro_batch_size * world_size()
    if denom <= 0:
        raise ValueError("micro_batch_size and world_size must be positive")
    if desired_global_batch % denom != 0:
        # Round up to next multiple so we never under-train.
        steps = (desired_global_batch + denom - 1) // denom
    else:
        steps = desired_global_batch // denom
    return max(1, steps)


def maybe_distributed_sampler(
    dataset,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> Optional["torch.utils.data.distributed.DistributedSampler"]:
    """Return a DistributedSampler when distributed, else None.

    Used by training loaders for map-style datasets. For streaming/iterable
    datasets, use `split_iterable_by_rank` below instead.
    """
    if not is_distributed():
        return None
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=world_size(),
        rank=rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def split_iterable_by_rank(iterable: Iterable, num_shards: Optional[int] = None, shard_index: Optional[int] = None) -> Iterable:
    """For HF streaming datasets: shard by rank using datasets.distributed
    when available, else manual round-robin fallback.

    Pre: `iterable` is an HF IterableDataset OR plain iterable.
    Post: returns the same type sharded so each rank sees a disjoint subset.
    """
    if not is_distributed() and num_shards is None:
        return iterable

    n = num_shards if num_shards is not None else world_size()
    idx = shard_index if shard_index is not None else rank()

    try:
        from datasets.distributed import split_dataset_by_node

        return split_dataset_by_node(iterable, rank=idx, world_size=n)
    except Exception:
        # Manual round-robin fallback for plain iterables.
        def _gen():
            for i, item in enumerate(iterable):
                if i % n == idx:
                    yield item

        return _gen()
