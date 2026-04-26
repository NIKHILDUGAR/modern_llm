"""Minimal training loop scaffold grounded in cross-entropy LM training (e.g., GPT family).

DDP-aware: when WORLD_SIZE > 1, the model is wrapped in DistributedDataParallel
and only rank 0 logs / saves checkpoints / runs the tqdm bar. When WORLD_SIZE
is unset or 1 the trainer behaves exactly as the previous single-GPU loop.
Checkpoints always store the unwrapped state_dict (no `module.` prefix), so
they remain interchangeable between single-GPU and multi-GPU runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import torch
from torch import nn, Tensor
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from modern_llm.config.train_config import TrainingConfig
from modern_llm.training.distributed import (
    get_device,
    init_distributed,
    is_distributed,
    is_main_process,
    reduce_mean,
    seed_everything,
    unwrap_model,
    world_size,
    wrap_ddp,
)
from modern_llm.quantization import get_quantization_payload, set_quantization_step
from modern_llm.utils.checkpointing import save_checkpoint
from modern_llm.utils.logging_utils import create_logger


@dataclass
class Trainer:
    """Causal LM trainer with gradient accumulation, AMP, and DDP."""

    model: nn.Module
    optimizer: Optimizer
    train_dataloader: Iterable
    config: TrainingConfig
    eval_dataloader: Optional[Iterable] = None
    lr_scheduler: Optional[_LRScheduler] = None

    device: torch.device = field(init=False)
    logger: logging.Logger = field(init=False)
    use_amp: bool = field(init=False)
    scaler: Optional[GradScaler] = field(init=False, default=None)
    global_step: int = field(init=False, default=0)
    micro_step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # 1. Bring up DDP (no-op when WORLD_SIZE<=1) and seed per rank.
        init_distributed()
        if self.config.seed is not None:
            seed_everything(self.config.seed)

        # 2. Place model on the per-rank device, then DDP + optional compile.
        # PyTorch recommends DDP-then-compile so inductor fuses through the
        # AllReduce hooks rather than graph-breaking at them.
        self.device = get_device()
        self.logger = create_logger(f"trainer.{self.config.run_name}")
        self.model.to(self.device)
        # NOTE: wrap_ddp returns the bare model when WORLD_SIZE<=1.
        self.model = wrap_ddp(self.model)
        if self.config.compile_model and hasattr(torch, "compile"):
            # python_reducer avoids the C++ DDP reducer's _broadcast_coalesced
            # call that dynamo cannot trace, which otherwise graph-breaks the
            # compiled region at every forward entry.
            from torch import _dynamo as torch_dynamo
            torch_dynamo.config.optimize_ddp = "python_reducer"
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

        # 3. AMP setup. bf16 needs no GradScaler; fp16 does.
        self.use_amp = self.config.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda"
        if self.config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.scaler = GradScaler()
        else:
            self.scaler = None

    # --------------------------------------------------------------- training

    def train(self) -> None:
        """Run the optimization loop.

        Pre:
            - model, optimizer, dataloaders initialized.
        Post:
            - checkpoints and logs emitted per configuration (rank 0 only).
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_steps = self.config.gradient_accumulation_steps
        max_steps = self.config.max_steps

        epoch = 0
        pbar = tqdm(
            total=max_steps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            disable=not is_main_process(),
        )
        try:
            while self.global_step < max_steps:
                # Set the epoch on a DistributedSampler so each epoch gets a
                # different shuffling deterministic per rank.
                self._set_sampler_epoch(self.train_dataloader, epoch)

                for batch in self.train_dataloader:
                    prev_step = self.global_step
                    loss = self._training_step(batch, accumulation_steps)
                    step_completed = self.global_step > prev_step

                    if step_completed:
                        pbar.update(1)

                        if (
                            self.config.log_every > 0
                            and self.global_step % self.config.log_every == 0
                            and is_main_process()
                        ):
                            self.logger.info(
                                "step=%d loss=%.4f lr=%.3e",
                                self.global_step,
                                loss,
                                self.optimizer.param_groups[0]["lr"],
                            )
                        if (
                            self.config.eval_every > 0
                            and self.eval_dataloader
                            and self.global_step > 0
                            and self.global_step % self.config.eval_every == 0
                        ):
                            metrics = self.evaluate()
                            if is_main_process():
                                self.logger.info(
                                    "eval step=%d loss=%.4f ppl=%.2f",
                                    self.global_step,
                                    metrics["loss"],
                                    metrics["perplexity"],
                                )
                        if (
                            self.config.save_every > 0
                            and self.global_step > 0
                            and self.global_step % self.config.save_every == 0
                        ):
                            self._save_checkpoint()

                    if self.global_step >= max_steps:
                        break
                if self.global_step >= max_steps:
                    break
                epoch += 1
        finally:
            pbar.close()

        self._save_checkpoint(suffix="final")

    def _training_step(self, batch: Dict[str, Tensor], accumulation_steps: int) -> float:
        if self.config.quantization is not None and self.config.quantization.enabled:
            set_quantization_step(unwrap_model(self.model), self.global_step)
        batch = self._move_batch_to_device(batch)
        raw_loss = self._forward_loss(batch)
        micro_loss = raw_loss / accumulation_steps

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(micro_loss).backward()
        else:
            micro_loss.backward()

        # Accumulate raw CE across micro-batches so the step-level log reports
        # mean(raw_loss), not the final micro-batch's (raw_loss / accum).
        self._accum_loss_sum = getattr(self, "_accum_loss_sum", 0.0) + float(raw_loss.detach().cpu())

        self.micro_step += 1
        step_completed = self.micro_step % accumulation_steps == 0
        reported_loss = float(raw_loss.detach().cpu())
        if step_completed:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.config.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            reported_loss = self._accum_loss_sum / accumulation_steps
            self._accum_loss_sum = 0.0
            self.global_step += 1

        return reported_loss

    def _forward_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        autocast_dtype = None
        if self.use_amp:
            autocast_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        with autocast(device_type="cuda", dtype=autocast_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            raise ValueError("Model must return a loss when labels are provided.")
        return loss

    # ------------------------------------------------------------ evaluation

    def evaluate(self) -> Dict[str, float]:
        if not self.eval_dataloader:
            return {"loss": float("nan"), "perplexity": float("nan")}
        self.model.eval()
        # fp32 accumulators: bf16 underflows/overflows in reduction easily.
        total_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        total_batches = torch.zeros(1, device=self.device, dtype=torch.float32)
        skipped = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._move_batch_to_device(batch)
                loss = self._forward_loss(batch).detach().float()
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                total_loss += loss
                total_batches += 1

        # Reduce so all ranks see identical eval numbers.
        if is_distributed():
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_batches, op=torch.distributed.ReduceOp.SUM)
            skipped_t = torch.tensor([skipped], device=self.device, dtype=torch.float32)
            torch.distributed.all_reduce(skipped_t, op=torch.distributed.ReduceOp.SUM)
            skipped = int(skipped_t.item())

        counted = float(total_batches.item())
        if counted == 0.0:
            if is_main_process():
                self.logger.warning("eval: every batch produced non-finite loss; skipped=%d", skipped)
            self.model.train()
            return {"loss": float("nan"), "perplexity": float("inf")}

        if skipped > 0 and is_main_process():
            self.logger.warning("eval: skipped %d non-finite batch losses", skipped)

        avg_loss = float((total_loss / total_batches).item())
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        self.model.train()
        return {"loss": avg_loss, "perplexity": perplexity}

    # ----------------------------------------------------------------- utils

    def _move_batch_to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @staticmethod
    def _set_sampler_epoch(loader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        # Only rank 0 writes to disk; other ranks block until it finishes
        # so subsequent reads (e.g., resume) see a complete file.
        if not is_main_process():
            from modern_llm.training.distributed import barrier
            barrier()
            return

        tag = suffix or f"step{self.global_step}"
        path = self.config.output_dir / f"{self.config.run_name}_{tag}.pt"

        # Always strip DDP/torch.compile wrappers before reading state_dict.
        model_for_state = unwrap_model(self.model)

        config_dict = None
        config_obj = getattr(model_for_state, "config", None)
        if config_obj is not None and hasattr(config_obj, "__dict__"):
            config_dict = {k: v for k, v in config_obj.__dict__.items() if not k.startswith("_")}
        quantization_payload = get_quantization_payload(model_for_state)

        checkpoint_metadata = {
            "step": self.global_step,
            "run_name": self.config.run_name,
            "config": config_dict,
        }
        if quantization_payload is not None:
            checkpoint_metadata["quantization"] = quantization_payload

        save_checkpoint(
            path,
            model_state=model_for_state.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            **checkpoint_metadata,
        )

        from modern_llm.training.distributed import barrier
        barrier()
