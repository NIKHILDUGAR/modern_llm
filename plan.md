# Plan: Beating GPT-2 (124M) With a Sub-75M Modern LM on 2x RTX 4090

**Author:** ML Research
**Date:** 2026-04-17 (revised 2026-04-20)
**Repo:** `/workspace/mnt/data_sda/lost+found/nikhil/modern_llm`
**Target:** A from-scratch decoder-only LM with < 75M parameters that strictly
outperforms GPT-2 small (124M) on the 16-benchmark suite listed in
`scripts/evaluation/` (gpqa, hle, ifbench_test, ifeval, mixeval_easy, mmlu_pro,
anli, bbq, commonsenseqa, coqa, glue, hellaswag, mmlu, squad_v2, gsm8k, sst2).

This revision supersedes previous versions. Headline differences vs the
pre-4090 plan:

1. **Target hardware is 2x RTX 4090 (24 GB each) over PCIe, DDP via
   `torchrun`** — not 1xH100. The launch layer is **GPU-count
   agnostic**: any subset of GPUs from 1 to N works by setting
   `CUDA_VISIBLE_DEVICES=<list>` plus `--nproc-per-node=<count>` on the
   wrapper. The trainer reads `LOCAL_RANK`/`WORLD_SIZE` from env.
2. **Custom 16k byte-level BPE tokenizer** at `tokenizers/cl_small_bpe_16k/`,
   trained on our own pretrain mix. Replaces the 100k cl100k tokenizer that
   was eating 51M of the 75M budget and forcing an 8-layer model. The 16k
   vocab with tied embeddings lets us run 18 layers at `d_model=512` with
   GQA 4:1.
3. **Pretrain is fed from pre-tokenized, packed uint32 shards on disk**,
   not from HF streaming. `data/tokenized/pretrain_mix/` holds 101 shards
   = **50,000,005,430 tokens = 12,207,032 windows of 4096 tokens** produced
   by `scripts/data/tokenize_pretrain.py`. The trainer mmap-reads through
   `PackedShardDataset` (`src/modern_llm/data/lm_datasets.py`), wired end-to-end
   via `packed_shards_dir=config.pretrain_packed_shards` from both the
   `scripts/run_pipeline.py` entrypoint and the `AlignmentPipeline._run_pretrain`
   multi-stage glue. Streaming is now a fallback, not the default.
4. **All HF dataset/model caches are redirected to `data/raw/` inside the
   repo**, via `scripts/data/migrate_hf_cache.sh` + env-var defaults set at
   process start.
5. **Pre-4090 `experiments/runs/gpu-full*` are archived under
   `experiments/runs/gpu-full_archive_pre-4090/`**. The tokenizer changed
   and the architecture shrank, so those checkpoints are structurally
   incompatible with the new model.
6. **Single-GPU launch shape is preserved.** The template
   `CUDA_VISIBLE_DEVICES=1 ./scripts/launch.sh --config gpu --stage all`
   still works. `run_pipeline.py` self-spawns under `torchrun --standalone
   --nproc_per_node=N` iff `--nproc-per-node > 1`.

---

## 0. Executive summary

- **Architecture (as built, `configs/lm_75m_2x4090.json`):** 18-layer
  decoder-only Transformer, `d_model=512`, `n_heads=8` (head_dim=64),
  `n_kv_heads=2` (GQA 4:1), SwiGLU FFN with `d_ff=1408` (~8/3 * d), RoPE,
  RMSNorm pre-norm with **QK-norm**, **tied** embeddings, `vocab_size=16000`,
  seq len 4096, **scaled embeddings**, **Megatron residual init scaling**,
  **PaLM z-loss (1e-4)**, `torch.compile` ON, attention sinks OFF,
  MoE OFF. **Total params: ~59.0M** (reported by `print_model_parameters`
  at run start). Well inside the 75M cap.
- **Tokenizer:** custom byte-level BPE, vocab 16 000 (incl. 7 specials),
  trained on the pretrain mix. Artifact at `tokenizers/cl_small_bpe_16k/`,
  loadable via `AutoTokenizer.from_pretrained`.
- **Token budget (pretrain).** Packed corpus on disk = 50.0B tokens.
  Config runs 196,000 opt-steps at global batch 32 seqs * 4096 tok =
  131,072 tok/step, which consumes 196k * 131k ≈ **25.7B pretrain
  tokens** (~0.5 epoch over the packed corpus; plenty of unique data
  remains for later stages/anneals). At 59M params that is ~435 tok/param —
  ~25x over Chinchilla-optimal, deliberately so (SmolLM-2 / MobileLLM
  "train to saturation" regime).
- **Stages:** (1) pretrain, (2) SFT, (3) DPO, (4) verifier training, with
  mid-training anneal and reasoning SFT as optional extensions. Gated by
  evals between stages.
- **Dataset storage:** HF caches live at `<repo>/data/raw/hf_cache` and
  `<repo>/data/raw/hf_home/hub`, seeded by `scripts/data/migrate_hf_cache.sh`
  and enforced by `src/modern_llm/utils/paths.apply_env_defaults()`.
- **Target hardware:** **2x RTX 4090 24 GB, PCIe Gen4, DDP** (bf16, no
  FSDP needed at this scale).

### The 5 big bets (why this beats GPT-2)

1. **Data quality, not model size.** GPT-2: 40 GB WebText. Us: ~25B
   packed tokens of FineWeb-Edu + DCLM-Baseline + OpenWebMath + Wikipedia
   + RedPajama StackExchange/arXiv. At 59M params this recipe should
   clear GPT-2-small by a wide margin on MMLU, HellaSwag, and CSQA.
2. **Depth over width (MobileLLM).** For sub-1B models, deeper-thinner
   wins the param/quality frontier. The 16k BPE makes 18 layers at
   d=512 feasible inside 75M; the cl100k path forced us down to 8 layers.
3. **Modern stack** — RoPE, GQA 4:1, SwiGLU, RMSNorm pre-norm, QK-norm,
   scaled embeddings, Megatron residual init, PaLM z-loss. Each worth a
   fraction of a nat; stacked they are 1-2 nats/token at 100M scale and
   buy bf16 stability for free.
4. **Instruction + preference + verifier distillation.** GPT-2 is a raw
   LM; our final checkpoint sees SFT (Tulu-3 + SmolTalk + OpenHermes +
   math + ifeval-like + no_robots + coqa), DPO (UltraFeedback-binarized),
   and a trained answer-correctness verifier. This alone decides IFEval,
   IFBench, MixEval-Easy, and CoQA.
5. **Pre-tokenized packed shards.** No per-step tokenizer/CPU bottleneck
   during pretrain; 4090s stay GPU-bound. Cross-shard window splicing via
   cumulative-offset binary search keeps the loader stateless and
   DDP-safe.

---

## 1. Parameter budget

### 1.1 Final architectural choice

Previous plan's "Option D" was 8 layers at d=512 because the 100k cl100k
tokenizer's embedding dominated the budget. Swapping to a **16k custom BPE
with tied embeddings** drops the embedding cost from 51.3M to 8.2M,
unlocking **18 layers** at the same d=512.

| Hyperparameter | Final Value | Rationale |
|---|---|---|
| `n_layers` | **18** | MobileLLM depth-over-width; more layers is the biggest lever under a param cap |
| `d_model` | **512** | `head_dim=64` sweet spot; divisible by 8 and 2 for GQA |
| `n_heads` (Q) | **8** | `d_model / head_dim = 512 / 64` |
| `n_kv_heads` | **2** | GQA 4:1 via `gqa_groups=2` — ~25% KV cache vs MHA |
| `head_dim` | 64 | Flash-Attn sweet spot |
| `d_ff` | **1408** | ~8/3 * d, rounded to 64x (LLaMA/SwiGLU convention) |
| `vocab_size` | **16 000** | 7 specials + 15 993 BPE merges from our own pretrain mix |
| `max_seq_len` | 4096 | RoPE handles this easily |
| `tie_embeddings` | **True** | Embed and LM head share weights; saves 8.2M |
| `rope_theta` | 10 000 (config default) | keep default since ctx is 4k |
| `dropout` | 0.0 pretrain / 0.1 SFT | Standard modern recipe |
| `scale_embeddings` | **True** | PaLM/LLaMA: multiply token embed by sqrt(d_model) |
| `residual_init_scale` | **True** | Megatron: scale residual proj by 1/sqrt(2*n_layers) |
| `z_loss_coef` | **1e-4** | PaLM §5.1 — stabilizes bf16 softmax |
| `use_qk_norm` | **True** | OLMo-2 / Chameleon — prevents attn-logit blowup |
| `use_attention_sinks` | False | breaks Flash fast-path; post-hoc enable for inference |
| `use_moe` | False | experts eat the param budget under a hard cap |
| `compile_model` | **True** | torch.compile on 4090 ~20-25% faster |

### 1.2 Per-block parameter count (d=512, d_ff=1408, n_heads=8, n_kv=2, head_dim=64)

| Component | Formula | Count |
|---|---|---|
| Q projection | d * d = 512*512 | 262 144 |
| K projection | d * (n_kv*head_dim) = 512*128 | 65 536 |
| V projection | d * (n_kv*head_dim) = 512*128 | 65 536 |
| O projection | d * d = 512*512 | 262 144 |
| SwiGLU (3 mats) | 3 * d * d_ff = 3*512*1408 | 2 162 688 |
| RMSNorm x2 | 2 * d | 1 024 |
| QK-norm x2 | 2 * head_dim | 128 |
| **Per block** | | **2 819 200** |

### 1.3 Totals

| Bucket | Count |
|---|---|
| 18 x decoder block | 50 745 600 |
| Final RMSNorm | 512 |
| Tied embedding (`vocab * d = 16000 * 512`) | 8 192 000 |
| **Total (tied, `vocab_size=16000`)** | **~58.94M** |

Runtime check: `print_model_parameters` at launch reports ~59.0M on
`configs/lm_75m_2x4090.json`, matching the analytic count above.

### 1.4 Contrast with GPT-2 small (124M)

| Component | GPT-2 small | Our model | Delta |
|---|---|---|---|
| `n_layers` | 12 | 18 | +50% |
| `d_model` | 768 | 512 | -33% |
| `d_ff` | 3072 | 1408 | -54% |
| Attention | MHA, 12 heads, learned pos | GQA 8/2, RoPE, QK-norm | KV cache ~6x smaller |
| Norm | LayerNorm (post-norm-ish) | RMSNorm + QK-norm | Faster + more stable bf16 |
| FFN | GELU(Wx)W' | SwiGLU (gated) | Better sample efficiency |
| Embedding | 50 257 * 768 = 38.6M | 16 000 * 512 = 8.2M (tied, shared with LM head) | -79% |
| Non-embed | ~85M | ~50.7M | -40% |
| Total | 124M | ~59M | -52% |

**The decisive savings come from the smaller vocabulary + tied
embeddings.** That is exactly what lets us go *deeper* than GPT-2 under
a *smaller* param cap, which is the whole theoretical underpinning of
beating it (deeper stacks at modest width dominate at <1B per MobileLLM).

---

## 2. Model design / new blocks

### 2.1 What the repo has (reuse as-is)

Confirmed in `src/modern_llm/models/`:

- `attention.py`: MultiHeadAttention with **RoPE**, **GQA**, optional
  attention sinks, **QK-norm** (per-head `RMSNorm(head_dim)` on Q and K
  before RoPE, applied on the manual-attention path to sink keys as well),
  SDPA/Flash fallback, head_dim validation.
- `layers.py`: **RMSNorm**, **SwiGLU**.
- `transformer.py`: `DecoderBlock` (pre-norm residual), `ModernDecoderLM`
  with:
  - tied embeddings (`tie_embeddings=True`),
  - **scaled embeddings** (`hidden *= sqrt(d_model)` when
    `scale_embeddings=True`),
  - **Megatron residual init scaling** (`attn.out_proj`/`ffn.proj`
    multiplied by `1/sqrt(2*n_layers)` after `_init_weights`),
  - **PaLM z-loss** folded into the cross-entropy when `z_loss_coef > 0`
    (mean over valid positions of `logsumexp(logits)^2`),
  - GPT-style `_init_weights`,
  - causal + padding attention bias construction.
- `moe.py`: Mixture-of-Experts FFN (unused — see §2.3).

All of the above are live in `configs/lm_75m_2x4090.json` with the
toggles flipped ON (`use_qk_norm`, `scale_embeddings`, `residual_init_scale`,
`z_loss_coef=1e-4`, `compile_model=true`).

### 2.2 What's still open in the training stack

| Feature | Status | Why | Where |
|---|---|---|---|
| **Sequence packing for SFT** | NOT DONE | 4k ctx on instruction data is mostly padding without packing | `data/instruction_datasets.py` |
| **uP / muP-lite init** | NOT DONE (soft) | Cerebras variant of Tensor Programs V — makes LR transfer more reliable across scaling sweeps | `model_config.py` + init |
| **Mid-training anneal stage** | NOT WIRED | Data recipe written (§3 stage 2) but no separate entrypoint; currently goes straight from pretrain to SFT | would need a new stage in `AlignmentPipeline` |

`torch.compile`, grad checkpointing, QK-norm, residual init scaling,
scaled embeddings, z-loss, streaming dataloader, and packed-shards
pretrain are all DONE and verified live in the running pipeline.

### 2.3 What we explicitly **don't** do

- **MoE.** `moe.py` exists and `use_moe=false` is set. Under a 75M hard
  param cap, experts eat the budget. Skip unless we relax the cap.
- **Mamba / SSM layers.** Hybrids (Jamba, Zamba) win at 7B+ but show no
  consistent lift at 100M on MMLU/HellaSwag.
- **Differential Attention (Ye et al. 2024).** Unproven at <1B.
- **nGPT normalization.** Too new, no public small-model results.
- **ALiBi.** Worse than RoPE at this scale.

---

## 3. Training-stage roadmap

Four stages are wired in the running pipeline (pretrain → SFT → DPO →
verifier); mid-training anneal and reasoning SFT are recipes we can
slot in between once the base run stabilizes.

- **Fast eval** (~1 h on 2x4090): sst2, hellaswag (500), commonsenseqa
  (500), mmlu (1/subject), anli, squad_v2 (200), gsm8k (50).
- **Full eval**: all 16 benchmarks, full sample size.

### Stage 1 — Pretrain

**Goal:** Strong base LM. Clear GPT-2 small on HellaSwag zero-shot (31.1%).

**Pretrain mix — packed to disk at `data/tokenized/pretrain_mix/` (50.0B
uint32 tokens, 101 shards, sharded at 500M tokens each).** Source keys
listed in `data/tokenized/pretrain_mix/index.json`:

| Source key in index | HF name | Text field |
|---|---|---|
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` (sample-350BT) | `text` |
| `dclm` | `mlfoundations/dclm-baseline-1.0` | `text` |
| `openwebmath` | `open-web-math/open-web-math` | `text` |
| `wikipedia` | `wikimedia/wikipedia` (20231101.en) | `text` |
| `rp-stackexchange` | `togethercomputer/RedPajama-Data-1T` / stackexchange | `text` |
| `rp-arxiv` | `togethercomputer/RedPajama-Data-1T` / arxiv | `text` |

(The Stack v2 smol was dropped because the HF account lacks gated access;
the code slice is not included in the current packed corpus. See §11.)

**Hyperparameters (DDP on 2x4090, world_size=2, from `configs/lm_75m_2x4090.json`):**

| HP | Value |
|---|---|
| Packed tokens available | 50.0B |
| Tokens actually consumed | ~25.7B (196k steps * 131k tok/step) |
| Seq len | 4096 (packed, no padding) |
| Micro-batch per GPU | 4 seqs (16k tok) |
| Global batch size (seqs) | 32 |
| Grad-accum | 32 / (4 * 2) = 4 |
| Global batch (tokens) | 32 * 4096 = **131 072 tok** |
| Steps | 196 000 |
| Warmup | 4 000 |
| Peak LR | 2e-3, cosine to 2e-4 (`min_lr_ratio=0.1`) |
| Optimizer | AdamW (defaults) |
| Precision | bf16 |
| Dropout | 0.0 |
| Z-loss | 1e-4 |
| Compile | On |
| Save every | 2 000 steps |
| Eval every | 5 000 steps |
| Log every | 50 steps |

**Launch (as currently running):**
```
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 \
    torchrun --standalone --nproc-per-node=2 \
    scripts/run_pipeline.py --config configs/lm_75m_2x4090.json --stage all
```
(Equivalent via wrapper: `./scripts/launch.sh --config configs/lm_75m_2x4090.json --stage all --nproc-per-node=2`.)

**Checkpoints:** rank-0-only via `unwrap_model()` (strips `module.` +
`_orig_mod.`); saved every 2000 steps. All ranks `barrier()` after save.

### Stage 2 — Mid-training / anneal (optional, not wired)

**Goal:** +1-3 MMLU, long-ctx robustness. 1B tokens, high-quality only.

Recipe kept around for when we slot it between pretrain and SFT:

| Dataset | HF name | Tokens | Weight |
|---|---|---|---|
| FineWeb-Edu top decile | `HuggingFaceFW/fineweb-edu` (score>=4) | 750M | 75% |
| Dolmino-mix-1124 | `allenai/dolmino-mix-1124` | 100M | 10% |
| OpenWebMath | `open-web-math/open-web-math` | 80M | 8% |
| TuluMath (CoT prime) | `allenai/tulu-3-sft-mixture` math subset | 30M | 3% |
| Wiki + books | `wikimedia/wikipedia` + Gutenberg subset | 40M | 4% |

- Peak LR = pretrain end-LR (2e-4); warmup 200, cosine to 2e-5.
- Batch/DDP same as pretrain; ~4000 steps.
- Would need a new stage hook in `AlignmentPipeline` (see TODO).

### Stage 3 — SFT

**Goal:** Instruction following. Unlocks IFEval, IFBench, MixEval-Easy.

Configured `sft_datasets` in `configs/lm_75m_2x4090.json`:

| Dataset | HF name |
|---|---|
| Tulu-3 SFT mixture | `allenai/tulu-3-sft-mixture` |
| SmolTalk | `HuggingFaceTB/smoltalk` |
| OpenHermes-2.5 | `teknium/OpenHermes-2.5` |
| MetaMathQA | `meta-math/MetaMathQA` |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` |
| IFEval-like-data | `argilla/ifeval-like-data` |
| NoRobots | `HuggingFaceH4/no_robots` |
| CoQA-train | `stanfordnlp/coqa` |

Current caveat: `AlignmentPipeline._run_sft` only threads `config.sft_dataset`
(a single name, defaulting to `allenai/tulu-3-sft-mixture`) into the
`InstructionDatasetConfig`. The multi-dataset list in `sft_datasets` is
a config-level declaration but is not yet consumed by SFT. See TODO.

- ChatML (`<|im_start|>user / assistant <|im_end|>`); loss masked on prompts.
- Global batch 32 (micro 4), seq 4096; steps 4 000.
- Peak LR 5e-5.

### Stage 4 — Preference optimization (DPO)

| Dataset | HF name | Rows |
|---|---|---|
| UltraFeedback (binarized) | `HuggingFaceH4/ultrafeedback_binarized` | 62k |

- DPO beta=0.05 (small models overfit 0.1).
- Peak LR 3e-6, global batch 16 (micro 2), 1 000 steps.
- Reference model = frozen SFT checkpoint.
- Contingency: if MMLU regresses >1 point, WiSE-FT soup
  (0.7 * theta_dpo + 0.3 * theta_sft).

HelpSteer2 + SkyworkReward are on the bench as next-preference-mix
additions (not currently in config).

### Stage 5 — Verifier training + optional reasoning SFT

**Part A — Verifier (wired):**

Runs after DPO: 2000 steps, LR 1e-4, global batch 32 (micro 4). Trains
a small answer-correctness head (`modern_llm.models.verifier`) for best-of-N
reranking at eval/inference time.

**Part B — Reasoning SFT + verifier-guided RS (not wired):**

| Dataset | HF name | Rows |
|---|---|---|
| OpenThoughts-114k | `open-thoughts/OpenThoughts-114k` | 114k |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | 860k |
| GSM8K train + self-generated rejection sample | `gsm8k` + RS | 7k + ~20k |

- Would be ~2000 steps from DPO checkpoint; LR 1e-5, warmup 100, cosine to 1e-6.
- Trigger Part B if GSM8K < 15% after DPO: sample 8 per train item, keep
  highest-scoring correct ones, retrain (STaR / V-STaR).

**Expected end-of-pipeline:** GSM8K 15-25 (GPT-2: ~0), IFEval >20,
MMLU 30-33, HellaSwag >36, MixEval-Easy >30.

---

## 4. Dataset storage policy

### 4.1 Canonical locations

All HF caches live **inside the repo** on the 1.8 TB data volume:

```
data/raw/
  hf_cache/        <- HF_DATASETS_CACHE (arrow shards)
  hf_home/         <- HF_HOME
  hf_home/hub/     <- HF_HUB_CACHE (model weights)
data/tokenized/
  pretrain_mix/    <- our packed uint32 shards (50.0B tokens / 101 shards)
    index.json
    shard_00000.bin ... shard_00100.bin
tokenizers/
  cl_small_bpe_16k/   <- custom 16k BPE tokenizer
```

### 4.2 How it's enforced

1. **`scripts/data/migrate_hf_cache.sh`** (idempotent) moves
   `~/.cache/huggingface/{datasets,hub}/*` into `data/raw/{hf_cache,hf_home/hub}/`
   and drops back-symlinks at the original locations. Safe to re-run.
2. **`scripts/launch.sh`** exports `HF_HOME`, `HF_DATASETS_CACHE`,
   `HF_HUB_CACHE` before execing python.
3. **`src/modern_llm/utils/paths.apply_env_defaults()`** is called at
   import time by `src/modern_llm/data/lm_datasets.py` (and the
   tokenizer-train script). It `setdefault`s the env vars so running
   `python scripts/run_pipeline.py ...` without the launch wrapper still
   redirects the cache.

### 4.3 Disk budget

| Artifact | Size |
|---|---|
| Packed pretrain uint32 shards | ~200 GB (50B * 4 bytes) |
| HF arrow caches for SFT / DPO / reasoning sources | ~30 GB |
| Tokenizer artifact | ~5 MB |
| Model checkpoints (pretrain every 2k x ~5 kept) | ~50 GB |
| **Total** | **~280 GB** — comfortably inside 1.8 TB |

---

## 5. Tokenizer decision (settled)

**Old plan:** keep `Xenova/text-embedding-ada-002` (cl100k, vocab 100 261).
**Current:** train our own **16k byte-level BPE** on the pretrain mix.

- cl100k's 100k vocab consumed 51M of the 75M budget at d=512 with tied
  embeddings, forcing `n_layers=8`. A domain-matched 16k BPE unlocks
  18 layers at d=512 inside the same cap.
- 16k vs 32k: chose 16k because (a) extra 32k merges are mostly
  code/math long-tail tokens with small weight in the final mix; (b) 16k
  saves 8.2M params vs 32k, buying 2-3 extra transformer layers; (c) we
  are not compression-bound at seq_len=4k.

**Artifact policy:** `tokenizers/cl_small_bpe_16k/` ships with every
release; both eval and downstream users need it since it is not on HF Hub.
`scripts/evaluation/_eval_common.py::DEFAULT_TOKENIZER` resolves in this
order: `$MODERN_LLM_TOKENIZER` env var > `tokenizers/cl_small_bpe_16k/` >
`Xenova/text-embedding-ada-002` (legacy fallback for archived pre-4090 checkpoints).

---

## 6. Distributed training: GPU-count agnostic DDP

- **Single source of DDP truth:** `src/modern_llm/training/distributed.py`.
- Entry points call `init_distributed()` (idempotent),
  `seed_everything(base_seed)` (each rank `base_seed + rank`), `get_device()`,
  `wrap_ddp(model)`. If `WORLD_SIZE<=1` `wrap_ddp` returns the bare model.
- Data: `maybe_distributed_sampler()` attaches a `DistributedSampler` when
  distributed, no-op otherwise. For streaming datasets we use
  `datasets.distributed.split_dataset_by_node()`.
- Packed shards: a single `PackedShardDataset` over all 101 shards is built;
  each rank reads disjoint windows via the `DistributedSampler` (no
  per-rank shard partitioning — all ranks mmap all shards). Memmaps are
  opened lazily on first access inside each worker process.
- Logging/tqdm gated on `is_main_process()`. `evaluate()` does `all_reduce`
  on loss/batches. Checkpoints saved on rank 0 only, `barrier()` before
  return. `unwrap_model()` strips both `module.` (DDP) and `_orig_mod.`
  (`torch.compile`) prefixes.

### 6.1 Launch layer

Two modes, same user-facing template:

1. **Single-GPU:**
   ```
   CUDA_VISIBLE_DEVICES=1 ./scripts/launch.sh --config gpu --stage all
   ```
2. **Multi-GPU DDP:**
   ```
   CUDA_VISIBLE_DEVICES=0,1 ./scripts/launch.sh \
       --config configs/lm_75m_2x4090.json --stage all --nproc-per-node 2
   ```

`scripts/launch.sh` exports HF cache env vars, sets NCCL PCIe defaults
(`NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_ASYNC_ERROR_HANDLING=1`),
NUMA-pins via `numactl`, then execs `python3 scripts/run_pipeline.py "$@"`.

`scripts/run_pipeline.py` adds `--nproc-per-node` (default 1). If `>1`
and we are not already under torchrun, it `os.execvp`s
`torchrun --standalone --nproc_per_node=N <argv>` (minus the wrapper flag).

### 6.2 Latent-regression fix: circular import

`src/modern_llm/training/__init__.py` used to eagerly re-export
`run_training`/`generate_text`, which deadlocked on `--stage all`
(alignment_pipeline → data → lm_datasets → training.distributed → training
→ train_lm → data partial). It is now a docstring-only module; callers
do `from modern_llm.training.train_lm import run_training` explicitly.

### 6.3 Latent-regression fix: model_config preservation in `run_training`

`train_lm.run_training` previously rebuilt `ModernLLMConfig` field-by-field
after reading vocab_size from the tokenizer, silently dropping
`use_qk_norm`, `z_loss_coef`, `scale_embeddings`, `residual_init_scale`,
etc. Now uses `dataclasses.replace(model_config, vocab_size=tokenizer.vocab_size)`
so every architectural toggle from the config survives.

### 6.4 Latent-regression fix: packed-shards in `AlignmentPipeline._run_pretrain`

Previously only `scripts/run_pipeline.py::run_pretrain` passed
`packed_shards_dir`. `AlignmentPipeline._run_pretrain` (used by
`--stage all`) silently fell back to HF streaming. Now both call sites
pass `packed_shards_dir=config.pretrain_packed_shards`, so `--stage all`
and `--stage pretrain` consume the same packed corpus.

---

## 7. Compute / time budget

### 7.1 Target hardware

- **Primary: 2x RTX 4090 24 GB, PCIe Gen4, DDP.**
  - bf16; Flash-Attn via SDPA; `torch.compile` ON.
  - Grad checkpointing available but not currently required — observed
    ~9.3 GB/GPU during compile warm-up at micro_batch=4, seq=4096.
- **Dev: RTX 3060 12 GB** for smoke/config validation
  (`configs/lm_max_rtx3060.json`, seq 1024, micro-batch 2, compile OFF).
- **Optional scale-up:** `--nproc-per-node 4` on a 4x4090 box Just Works.

### 7.2 Throughput estimate

Per-GPU ~75 TFLOP/s sustained bf16 on 4090 (conservative).

- FLOPs/token ~= 6 * 59e6 ~= 3.5e8.
- 75 TFLOP/s / 3.5e8 ~= ~215k tok/s/GPU realistic. With DDP overhead
  budget ~350-400k tok/s aggregate on 2 GPUs.
- 25.7B tokens / ~375k tok/s = **~19 hours pretrain** (best case).

Actual timing will be updated from M17 logs as they land.

### 7.3 Wall-clock plan (2x RTX 4090, current config)

| Stage | Steps / Tokens | Time est. |
|---|---|---|
| Pretrain | 196k / ~25.7B | ~19-28 h |
| SFT | 4k / ~0.5B | ~1-2 h |
| DPO | 1k / ~30-50M | ~0.3-0.5 h |
| Verifier | 2k / ~0.2B | ~0.5 h |
| Evals (full, per stage x 2) | — | ~4-6 h total |
| **Total** | | **~26-37 h wall-clock** |

Add a 2x safety margin for restarts, debug, failed runs: **~3-4 days**.

---

## 8. Evaluation plan

Targets unchanged structurally. Current baseline numbers (from
`experiments/results/eval_all_summary.json`, evaluated on an earlier
SFT checkpoint — pre-packed-shards / pre-QK-norm, for reference only):

| Benchmark | GPT-2 small (124M) | Earlier SFT ckpt | Our target |
|---|---|---|---|
| HellaSwag (acc_norm, 10-shot) | 31.1 | 27.8 | **>34.0** |
| MMLU (5-shot) | 25.9 | 23.8 | **>28.0** |
| CommonsenseQA (0-shot) | ~19.5 | 22.0 | **>30.0** |
| ANLI r1 | ~33.1 | 32.8 | **>34.0** |
| BBQ | ~50 | n/a | **>52** |
| SQuAD v2 (F1) | ~5-10 | 19.8 | **>25** |
| CoQA (F1) | ~10-15 | 0.8 | **>40** |
| GLUE (avg, 0-shot) | ~35 | n/a | **>40** |
| GSM8K (EM, 8-shot CoT) | ~0.5 | not run | **>10** |
| SST-2 (0-shot) | ~50 | not run | **>65** |
| IFEval (strict-instr) | <5 | not run | **>20** |
| IFBench test | <5 | not run | **>10** |
| MixEval-Easy | ~10 | not run | **>30** |
| MMLU-Pro (5-shot) | ~11 | 11.3 | **>14** |
| GPQA (0-shot) | ~25 | 24.8 | **>26** (tie ok) |
| HLE | sub-random | n/a | **match or beat random** |

The earlier-checkpoint numbers are the pre-packed-shards baseline; the
current M17 run is the first time the full architecture (QK-norm, z-loss,
scaled embeddings, residual init, 50B packed tokens) trains end-to-end.

"Beats GPT-2" = **win on >=13 of 16**, including all instruction and
reasoning benchmarks.

Eval infrastructure (`scripts/evaluation/`) is complete: `eval_all.py`
drives all 16 task scripts and writes per-task `*_metrics.json` +
`eval_all_summary.json`. A sweep runner (`run_eval_sweep.py`) exists
for multi-checkpoint comparisons.

---

## 9. Risks and contingencies

| Risk | Probability | Mitigation |
|---|---|---|
| NCCL hang over PCIe on 2x4090 | Medium | `NCCL_P2P_DISABLE=1` default; resolved via smoke — 4090 driver hw-disables P2P regardless, NCCL falls back to SHM. |
| Activations OOM at seq=4096 on 24 GB 4090 | Low (observed ~9.3 GB/GPU during compile) | Grad checkpointing available; drop to micro-batch 2 if needed; seq 2048 as last resort. |
| `torch.compile` recompilation on grad-checkpointed blocks | Low | Smoke showed zero recompiles with compile ON + grad-ckpt ON; stable steady-state ~4.2 step/s. |
| 25.7B tokens insufficient to beat GPT-2 on MMLU | Medium | Extend pretrain by rolling through the remaining ~24B packed tokens; or swap in the anneal stage. |
| DPO regresses MMLU/HellaSwag | Medium | WiSE-FT soup with SFT weights (§3 stage 4). |
| GSM8K stuck at ~0 | High | Trigger verifier-guided RS (Stage 5B); add OpenMathInstruct-2 to SFT; self-consistency at eval. |
| Checkpoint saved on 2xGPU fails to load on 1xGPU | Low | `unwrap_model()` strips `module.`/`_orig_mod.` on save. |

---

## 10. Milestones / ordered task list

### Phase A — Plumbing (DONE)

1. **M1 DONE**: HF cache redirect (`paths.py`, `apply_env_defaults()`,
   `cache_dir_for_datasets()` threaded through loaders; `migrate_hf_cache.sh`).
2. **M2 DONE**: GPU-count-agnostic DDP (`distributed.py`; all trainers
   refactored; rank-0 save; `barrier()` sync; DistributedSampler /
   `split_dataset_by_node`; self-spawn `run_pipeline.py`; `launch.sh` with
   NUMA + NCCL PCIe defaults).
3. **M3 DONE**: `scripts/archive_old_runs.sh` (idempotent).
4. **M4 DONE**: `scripts/data/train_tokenizer.py` (16k byte-level BPE) +
   artifact at `tokenizers/cl_small_bpe_16k/`.
5. **M5 DONE**: `scripts/evaluation/_eval_common.py::DEFAULT_TOKENIZER`
   prefers 16k BPE when present, falls back to cl100k.
6. **M6 DONE**: `configs/lm_75m_2x4090.json` with final 18L/d=512
   architecture, world_size 2, bf16, compile ON, all modern toggles ON.
7. **M7 DONE**: QK-norm (per-head `RMSNorm(head_dim)` on Q and K before
   RoPE, applied to sink_k on manual path) in `attention.py`.
8. **M8 DONE**: Residual init scaling + scaled embeddings + PaLM z-loss
   in `transformer.py`; threaded through `ModernLLMConfig` and
   `PipelineConfig`.
9. **M9 DONE**: `torch.compile` plumbed end-to-end with capability-based
   auto-resolution.
10. **M10 DONE**: Attention sinks OFF for training in config
    (`use_attention_sinks: false`).

### Phase B — Data (DONE)

11. **M11 DONE**: first tokenizer train on 16k BPE.
12. **M12 DONE**: `scripts/data/ensure_dataset.py` helper.
13. **M13 DONE**: `download_pretrain_mix.py` — corpus downloaded.
14. **M14 DONE**: `tokenize_pretrain.py` — 101 packed uint32 shards
    at `data/tokenized/pretrain_mix/` (50.0B tokens, 12.2M windows of 4096).
15. **M15 DONE**: `download_sft_mix.py` + `tokenize_sft.py`.
16. **M16 DONE**: `download_dpo_mix.py` + `download_reasoning_mix.py`.

### Phase B' — Training-loop wiring (DONE, was the latent regression)

16a. **M16a DONE**: `PackedShardDataset` in `lm_datasets.py` (memmap-per-worker,
     cross-shard splicing via cumulative-offset binary search) +
     `load_packed_pretrain_dataset(data_dir, seq_len)` helper.
16b. **M16b DONE**: `train_lm.run_training(packed_shards_dir=...)` path
     — bypasses HF streaming when set; uses WikiText-2 validation as
     stable eval signal.
16c. **M16c DONE**: Both call sites wired — `scripts/run_pipeline.py::run_pretrain`
     AND `AlignmentPipeline._run_pretrain` pass
     `packed_shards_dir=config.pretrain_packed_shards`.
16d. **M16d DONE**: `model_config` preservation via `dataclasses.replace`
     (fixes silent drop of `use_qk_norm`, `z_loss_coef`, `scale_embeddings`,
     `residual_init_scale` etc.).
16e. **M16e DONE**: Circular-import fix in `training/__init__.py`
     (now a docstring-only module; callers import `run_training` explicitly).

### Phase C — Stage runs (IN FLIGHT)

17. **M17 IN PROGRESS**: Pretrain, 2x4090. `--stage all` pipeline live
    (pid 333768, log `logs/m17_all.log`) — pretrain → SFT → DPO → verifier.
18. **M18**: Mid-training anneal + fast eval (optional; not wired yet).
19. **M19**: SFT + **full eval** (gate).
20. **M20**: DPO + fast eval + IFEval + MixEval-Easy.
21. **M21**: Reasoning SFT (+ optional 5B verifier RS) — not wired.
22. **M22**: Full eval on final checkpoint.

### Phase D — Analysis & ship

23. **M23**: `experiments/results/final_report.md` — per-stage deltas
    vs GPT-2 small, GPT-2 medium, SmolLM-2-135M, across all 16 evals.
24. **M24**: Tag final checkpoint at
    `experiments/runs/lm-75m-2x4090/verifier_final.pt` (or
    `reasoning_final.pt` if Stage 5B triggers).

---

## 11. Open questions / known gaps

### Resolved

1. **Grad-checkpointing + `torch.compile` interaction.** Resolved: no
   per-step recompilation (smoke with `TORCH_LOGS=recompiles` emitted zero
   events).
2. **NCCL + `NCCL_P2P_DISABLE` on 2x4090.** Resolved: 4090 driver
   hw-disables P2P over PCIe regardless; NCCL falls back to SHM. The env
   var is belt-and-suspenders.
3. **16k tokenizer artifact ships with the model.** Yes — required for
   both eval and downstream use since it is not on HF Hub.
4. **Packed pretrain shards wired into the trainer.** Yes — both
   entrypoints now pass `packed_shards_dir`.

### Still open

5. **The Stack v2 smol access.** Gated, access not yet granted for the
   current HF account. The code slice is NOT in the current packed mix.
   Action: either request+receive Stack v2 smol access, or bake
   `bigcode/the-stack-smol` into a future repack.
6. **SFT multi-dataset consumption.** Config's `sft_datasets` list is
   declarative but `AlignmentPipeline._run_sft` still passes only
   `config.sft_dataset` (single). Need a multi-dataset SFT loader (or
   a pre-packed SFT corpus analogous to pretrain) before all 8 listed
   sources actually train the model.
7. **No separate mid-training anneal stage** in `AlignmentPipeline`.
   Recipe exists on paper (§3 stage 2) but there is no entrypoint.
8. **Reasoning-SFT + verifier-guided RS** is recipe-only; not wired.

---

## TODO / Future Improvements

Concrete next steps for the sub-75M LM → beat-GPT-2 effort, ordered by
expected impact-per-effort. Each item: what, why at this scale, rough
effort/impact estimate.

### Architecture

- **Complete muP-lite init.** Implement the Cerebras muP-lite variant
  (input embedding std = 1/sqrt(d_model), hidden weights std fixed,
  output embedding std = 1/sqrt(d_model), LR per layer scaled). Today
  we do scaled embeddings + Megatron residual init but not the full muP
  treatment. **Why:** lets the LR found at d=256/n_layers=8 transfer to
  the production d=512/n_layers=18 without resweeping. Crucial if we
  later push to 22 layers (§1.3 headroom). **Effort:** 1 day. **Impact:**
  high for future scaling, low on this run.
- **Push depth to 22 layers** (keeps total <= 70.4M). Once throughput
  numbers from M17 land, verify 22L fits in 24 GB at micro_batch=4 and
  re-run. **Why:** MobileLLM's depth-over-width frontier says +22% depth
  under the same param cap buys ~0.5-1% on MMLU and ~1% on HellaSwag at
  this scale. **Effort:** config change + one shakedown. **Impact:** medium.
- **Untied output head, small softmax temperature.** Tied embeddings
  save 8.2M; an untied head with a tiny `d_model → vocab` low-rank
  projection (rank 256) keeps the savings but lets the head specialize.
  **Why:** tied embeddings are a known quality ceiling at sub-100M;
  untying the head alone lifts perplexity measurably (see SmolLM-2).
  **Effort:** 0.5 day. **Impact:** medium.
- **Longer context via RoPE-theta bump + YaRN staging.** `rope_theta` is
  still the 10k default in `ModernLLMConfig`. Bump to 500k (or schedule:
  10k during pretrain first half, 500k during second half + anneal), and
  add a short YaRN-style extrapolation fine-tune at 8k ctx. **Why:**
  decouples long-context ability from compute. **Effort:** config + 1B
  anneal tokens. **Impact:** medium for CoQA/SQuAD; high for any
  long-ctx evals we add later.
- **Revisit MoE only if the param cap relaxes.** `moe.py` is ready.
  Under a 100M or "75M active, 200M total" budget, switch 4 of 18 FFNs
  to 8-expert top-2 MoE and we get GPT-2-medium quality at GPT-2-small
  active params. **Why:** sparsity is the only way to meaningfully beat
  the FLOPs/quality frontier at this size. **Effort:** 2-3 days
  (router + load-balancing loss + eval). **Impact:** high if cap relaxes,
  N/A otherwise.

### Data

- **Ingest a code slice.** The current packed mix has NO code; that is
  the single biggest quality gap vs a SmolLM-style recipe. Either (a)
  get Stack v2 smol access, or (b) pack `bigcode/the-stack-smol` +
  OpenCoder-1.5T subset into a new `data/tokenized/pretrain_code/` dir
  and mix at 10% during a second pretrain phase. **Why:** code boosts
  reasoning/MMLU-STEM even when eval has no coding tasks (cross-domain
  transfer is well-documented). **Effort:** 1 day download + tokenize;
  1 day trainer mix-weight plumbing. **Impact:** high.
- **Weighted sampling across the 101 shards.** Today `DistributedSampler`
  over `PackedShardDataset` treats every window as equally likely. The
  index shows rp-arxiv ended up over-weighted (~45% of total tokens vs
  the 2% intended). Add a per-window source_id and an up-front weighted
  sampler that hits target domain ratios. **Why:** the current
  rp-arxiv dominance biases the prior toward dense scientific prose and
  hurts casual-English eval (HellaSwag, CSQA). **Effort:** 1 day.
  **Impact:** medium-high.
- **Near-dedup on FineWeb-Edu.** Even curated sets have ~5-10%
  near-duplicate leakage. Run MinHash-LSH at shingle size 5, Jaccard 0.8
  as a preprocessing pass before a re-pack. **Why:** duplicates inflate
  train loss without improving generalization (Lee et al. 2021).
  **Effort:** 1-2 days on a CPU box. **Impact:** medium.
- **Add synthetic-reasoning corpus to pretrain tail.** Dump 200-500M
  tokens of Cosmopedia-v2 or distilled OpenMathReasoning into the last
  10% of pretrain (a de-facto mini-anneal on the main run). **Why:**
  hits GSM8K directly, which is the hardest benchmark to move post-hoc.
  **Effort:** 1 day. **Impact:** high for GSM8K.
- **Context-length curriculum.** Start pretrain at seq_len=2048 for the
  first ~30% of tokens (higher throughput), then restart at 4096. **Why:**
  ~1.4x faster early pretrain without harming downstream perplexity —
  we do not learn much about 4k-ctx dependencies in the first 5B tokens
  anyway. **Effort:** trainer change to two-phase LR schedule.
  **Impact:** medium (throughput).

### Training

- **Schedule-Free AdamW or Shampoo.** Replace the current AdamW + cosine
  with either Schedule-Free AdamW (Defazio et al. 2024 — no decay
  schedule, iterate averaging) or Distributed Shampoo. **Why:** at
  sub-100M, Shampoo has shown 10-25% fewer tokens to reach target loss;
  Schedule-Free removes one tuning knob. **Effort:** 1 day (Shampoo has a
  torch implementation); 2 days for a proper bake-off. **Impact:**
  medium-high.
- **EMA of weights.** Maintain a 0.999-momentum EMA of the model during
  pretrain, use it for eval/SFT init. **Why:** at 59M + bf16, late-pretrain
  loss jitter is visible; EMA cuts eval variance and usually gives a
  0.3-0.5% lift on MMLU. **Effort:** 2 hours. **Impact:** low-medium.
- **Double the global batch via grad accum.** Current global batch is
  131k tokens; LLaMA-3 norm at this scale is 1-4M tokens. Bump
  `pretrain_batch_size` from 32 to 64 seqs (accum 8 instead of 4),
  halve the LR if divergence shows up. **Why:** large batch +
  proportional LR is strictly better at sub-100M for stability and
  terminal loss; we are GPU-bound, not memory-bound. **Effort:**
  config change + smoke. **Impact:** medium.
- **Stochastic depth during late pretrain.** Drop residual blocks with
  p=0.05 linearly scaled by depth for the final 10% of pretrain. **Why:**
  cheap regularizer at this scale; SmolLM-2 ablation shows 0.2-0.5% lift.
  **Effort:** 0.5 day. **Impact:** low.
- **Re-enable grad checkpointing only on the longest-context SFT.** For
  pretrain at 9.3 GB/GPU it's unnecessary and costs ~35% throughput.
  **Why:** the current plan carries it "for safety"; a clean audit
  removes it. **Effort:** trivial config. **Impact:** medium (throughput).

### Alignment

- **Swap DPO for SimPO or ORPO.** DPO's reference-model-anchored loss
  over-constrains small models. SimPO (reference-free, length-normalized)
  and ORPO (single-stage SFT + preference) both show 1-3 pt gains on
  sub-1B chat quality benchmarks vs DPO. **Why:** DPO beta=0.05 is a
  workaround for small-model over-fit; SimPO removes the need. **Effort:**
  1-2 days (new `train_simpo.py`, reuse dataloader). **Impact:** high for
  IFEval / MixEval-Easy.
- **KTO on unpaired feedback.** `HuggingFaceH4/ultrafeedback_binarized`
  wastes information; KTO (Kahneman-Tversky Optimization) trains from
  pointwise "good / bad" signals, letting us fold in HelpSteer2's
  Likert ratings directly. **Why:** 2-5x more usable signal per example
  at this scale. **Effort:** 2 days. **Impact:** medium.
- **Rejection-sampling fine-tune (RFT) before DPO.** After SFT, sample
  8 candidates per prompt on a GSM8K-like set, keep the correct
  highest-logprob one, and do a 500-step SFT pass on those. **Why:**
  distills the verifier's signal into the policy before DPO even runs;
  most of the reasoning lift at this scale comes from RFT, not from DPO.
  **Effort:** 2-3 days (depends on verifier readiness). **Impact:** high
  for GSM8K.
- **Constitutional-AI-lite self-critique pass.** After SFT, generate
  responses, let the model rewrite them against a 5-principle rubric,
  then SFT on the rewrites. **Why:** IFBench is largely about
  principle adherence; CAI is the cheapest way to hit it. **Effort:**
  1 day for the script, 0.5 day compute. **Impact:** medium for IFBench.
- **Train a dense reward model on UltraFeedback + HelpSteer2.** Our
  current "verifier" is a correctness classifier for math. A proper
  reward model lets us do PPO/GRPO later and also gates DPO data
  quality. **Why:** sub-100M RL is finally tractable with GRPO;
  prerequisite is a good RM. **Effort:** 2-3 days. **Impact:** high
  if we ever want RL.

### Evaluation

- **Held-out perplexity on FineWeb-Edu tail + Paloma.** Right now we
  only have task accuracies; add per-domain perplexity (web, code, math,
  wiki) for loss-curve monitoring. **Why:** task metrics are high-variance
  at 59M; perplexity on a held-out slice gives ~10x lower-variance signal
  for mid-run decisions. **Effort:** 0.5 day. **Impact:** high
  (decision-making).
- **Integrate EleutherAI lm-eval-harness** for BoolQ, PIQA, ARC-Easy,
  ARC-Challenge, WinoGrande, OpenBookQA. **Why:** standard sub-1B
  reporting suite; gives us direct apples-to-apples vs SmolLM-2 and
  Pythia-70M/160M. Missing from `experiments/results/` today. **Effort:**
  1 day. **Impact:** medium (communication/comparability).
- **Add a GSM8K eval runner.** `scripts/evaluation/` has no
  `eval_gsm8k.py`; we need 8-shot CoT + self-consistency@8 numbers
  before we can gate Stage 5B. **Why:** can't measure what we're
  optimizing. **Effort:** 0.5 day. **Impact:** prerequisite for
  reasoning work.
- **Contamination audit.** Check MMLU/GSM8K train splits against the
  packed pretrain corpus via 13-gram exact match; remove any
  matching shards before re-packing. **Why:** sub-1B models at 25B
  tokens are near the memorization edge; 0.1% contamination easily
  swings a benchmark by 1-2%. **Effort:** 1 day. **Impact:** medium
  (trustworthiness).
- **Stage-gain plots.** `experiments/results/stage_gains.md` exists but
  is stale; rebuild it as a programmatic dump of pretrain→SFT→DPO→verifier
  deltas per benchmark. **Why:** one of the strongest artifacts to
  justify each stage. **Effort:** 0.5 day. **Impact:** medium.

### Inference

- **Speculative decoding with a 15M draft model.** Train a 15M-param
  (d=256, L=8) draft on the same tokenizer + same packed shards for
  ~50B tokens (cheap on 1 GPU). At inference, accept its tokens under
  the main model's distribution. **Why:** 2-3x throughput at near-zero
  quality cost; matters for the eval sweep (16 benchmarks * hundreds of
  completions each). **Effort:** 3-4 days (separate training +
  integration). **Impact:** high on eval wallclock.
- **KV-cache quantization (int8).** Swap the KV cache to int8 with
  per-channel scale during generation. **Why:** 4090 inference is KV-bound
  at seq_len=4096 + batch>1; int8 KV fits 4x the batch. **Effort:** 1 day
  if we piggyback on `transformers` infra, 3 days from scratch.
  **Impact:** medium.
- **bf16 → int4 weight quantization at serve time** (GPTQ or AWQ).
  **Why:** 59M model fits in ~30 MB at int4 — enables CPU-only demo and
  4090 batch-size headroom. **Effort:** 1-2 days. **Impact:** low for
  evals, high for deployment story.
- **Pre-compute `torch.compile` cache artifact.** Compile warm-up costs
  ~90 s per run; `torch._inductor.codecache` persistence would shave that
  on eval sweeps. **Why:** 16 evals * 4 stages = 64 compile warm-ups
  today. **Effort:** 0.5 day. **Impact:** low-medium (wallclock).

---

**End of plan.**
