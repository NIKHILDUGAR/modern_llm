# Plan: Beating GPT-2 (124M) With a Sub-75M Modern LM on 2x RTX 4090

**Author:** ML Research
**Date:** 2026-04-17
**Repo:** `/workspace/mnt/data_sda/lost+found/nikhil/modern_llm`
**Target:** A from-scratch decoder-only LM with < 75M parameters that strictly
outperforms GPT-2 small (124M) on the 16-benchmark suite listed in
`scripts/evaluation/` (gpqa, hle, ifbench_test, ifeval, mixeval_easy, mmlu_pro,
anli, bbq, commonsenseqa, coqa, glue, hellaswag, mmlu, squad_v2, gsm8k, sst2).

This revision supersedes the previous plan. Major differences:

1. **Target hardware is 2x RTX 4090 (24 GB each) over PCIe, DDP via
   `torchrun`** — not 1xH100 as assumed before. The launch layer is explicitly
   **GPU-count agnostic**: any subset of GPUs from 1 to N works by setting
   `CUDA_VISIBLE_DEVICES=<list>` plus `--nproc-per-node=<count>` on the
   wrapper. The trainer reads `LOCAL_RANK`/`WORLD_SIZE` from env.
2. **Custom 16k byte-level BPE tokenizer** trained on our own pretrain mix
   (FineWeb-Edu + The Stack v2 smol + OpenWebMath). Replaces the 100k
   cl100k tokenizer that was eating 51M of the 75M budget and forcing an
   8-layer model. The 16k vocab with tied embeddings lets us run 18 layers
   at `d_model=512` with GQA 4:1.
3. **All HF dataset/model caches are redirected to `data/raw/` inside the
   repo**, via `scripts/data/migrate_hf_cache.sh` + env-var defaults set at
   process start. This moves HF state off the overlay FS onto the 1.8 TB
   data volume.
4. **Pre-4090 `experiments/runs/gpu-full*` are archived under
   `experiments/runs/gpu-full_archive_pre-4090/`** because (a) the tokenizer
   changed, (b) the architecture shrank, so those checkpoints are
   structurally incompatible with the new model. Eval-time loaders fall
   back to the cl100k tokenizer for those archived runs.
5. **Single-GPU launch shape is preserved.** The user's preferred template
   `CUDA_VISIBLE_DEVICES=1 ./scripts/launch.sh --config gpu --stage all`
   still works. `scripts/launch.sh` adds NUMA pinning + HF-cache exports +
   NCCL defaults for PCIe consumer GPUs, then execs
   `python3 scripts/run_pipeline.py`. `run_pipeline.py` self-spawns under
   `torchrun --standalone --nproc_per_node=N` iff `--nproc-per-node > 1`.

---

## 0. Executive summary

- **Architecture:** 18-layer decoder-only Transformer, `d_model=512`,
  `n_heads=8` (head_dim=64), `n_kv_heads=2` (GQA 4:1), SwiGLU FFN with
  `d_ff=1408` (~8/3 * d), RoPE (theta=500k), RMSNorm pre-norm with
  QK-norm, **tied** embeddings, `vocab_size=16000`, seq len 4096.
  **Total params: ~67.5M** (non-embed ~50.7M, embed ~8.4M shared input/output).
- **Tokenizer:** custom byte-level BPE, vocab 16 000 (incl. 7 specials),
  trained on FineWeb-Edu sample-10BT + The Stack v2 smol + OpenWebMath.
  Artifact at `tokenizers/cl_small_bpe_16k/`, loadable via
  `AutoTokenizer.from_pretrained`.
- **Token budget:** ~20B pretrain + 1B anneal + 0.6B SFT + 0.1B DPO +
  0.3B reasoning-SFT = **~22B training tokens**. ~330 tokens/param for
  pretrain — ~20x over Chinchilla-optimal and deliberately so (SmolLM-2 /
  MobileLLM).
- **Stages:** (1) pretrain, (2) mid-training anneal, (3) SFT, (4) DPO,
  (5) reasoning SFT + optional verifier-guided rejection sampling.
  Gated by evals between stages.
- **Dataset storage:** HF caches live at `<repo>/data/raw/hf_cache` and
  `<repo>/data/raw/hf_home/hub`, seeded by `scripts/data/migrate_hf_cache.sh`
  and enforced by `src/modern_llm/utils/paths.apply_env_defaults()`.
- **Target hardware:** **2x RTX 4090 24 GB, PCIe Gen4, DDP** (bf16, no
  FSDP needed at this scale). Wall-clock ~5 days at 2x, ~9 days at 1x.

### The 5 big bets (why this beats GPT-2)

1. **Data quality, not model size.** GPT-2: 40 GB WebText. Us: ~20B
   tokens of FineWeb-Edu + DCLM-Baseline + code + math. At 75M params
   this recipe dominates GPT-2-small numbers by ~30% relative on MMLU.
2. **Depth over width (MobileLLM).** For sub-1B models, deeper-thinner
   wins the param/quality frontier. The 16k BPE makes 18 layers at
   d=512 feasible inside 75M; the cl100k path forced us down to 8 layers.
3. **Modern stack** — RoPE (theta=500k), GQA 4:1, SwiGLU, RMSNorm
   pre-norm, QK-norm. Each worth a fraction of a nat; stacked they are
   1-2 nats/token at 100M scale.
4. **Instruction + preference + reasoning distillation.** GPT-2 is a raw
   LM; our final checkpoint sees SFT (Tulu-3 + SmolTalk + math), DPO
   (UltraFeedback + HelpSteer2), and reasoning-SFT (OpenThoughts +
   NuminaMath + GSM8K-aug). This alone decides IFEval, IFBench,
   MixEval-Easy, and GSM8K.
5. **Mid-training anneal on high-quality / long-context data.** A 1B
   anneal at 10x lower LR lifts MMLU 2-5 points at this scale for free.

---

## 1. Parameter budget

### 1.1 Final architectural choice (Option E — new default)

The previous plan's Option D was 8 layers at d=512 because the 100k
cl100k tokenizer's embedding dominated the budget. Swapping to a **16k
custom BPE with tied embeddings** drops the embedding cost from 51.3M
to 8.4M, unlocking **18 layers** at the same d=512.

| Hyperparameter | Final Value | Rationale |
|---|---|---|
| `n_layers` | **18** | MobileLLM depth-over-width; more layers is the biggest lever under a param cap |
| `d_model` | **512** | `head_dim=64` sweet spot; divisible by 8 and 2 for GQA |
| `n_heads` (Q) | **8** | `d_model / head_dim = 512 / 64` |
| `n_kv_heads` | **2** | GQA 4:1 — ~25% KV cache vs MHA, minor quality hit |
| `head_dim` | 64 | Flash-Attn sweet spot |
| `d_ff` | **1408** | ~8/3 * d, rounded to 64x (LLaMA/SwiGLU convention) |
| `vocab_size` | **16 000** | 7 specials + 15 993 BPE merges from our own pretrain mix |
| `max_seq_len` | 4096 | RoPE theta=500k supports this easily |
| `tie_embeddings` | **True** | Embed and LM head share weights; saves 8.4M |
| `rope_theta` | 500 000 | LLaMA-3 convention; extrapolates cleanly past 4k |
| `dropout` | 0.0 pretrain / 0.1 SFT | Standard modern recipe |
| `init` | std=0.02, residual proj scaled 1/sqrt(2*n_layers) | GPT-2 + Megatron residual scaling |

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

With `vocab_size=16384` (after specials and rounding up the effective
vocab from `tokenizers.Tokenizer` output) the embedding is 8 388 608 and
total is **~59.14M**. Either way, **well under 75M with room to spare**.

Headroom: if we want to push toward the cap we can go to **22 layers**
(+11.3M, total ~70.4M). The current choice of 18 is conservative so that
the 4090 memory budget for 4k ctx + bf16 activations still fits comfortably
with micro-batch 2 per GPU. This is the primary tuning lever after the
first pretrain run sees throughput numbers.

### 1.4 Contrast with GPT-2 small (124M)

| Component | GPT-2 small | Our model | Delta |
|---|---|---|---|
| `n_layers` | 12 | 18 | +50% |
| `d_model` | 768 | 512 | -33% |
| `d_ff` | 3072 | 1408 | -54% |
| Attention | MHA, 12 heads, learned pos | GQA 8/2, RoPE | KV cache ~6x smaller |
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

### 2.1 What the repo already has (reuse as-is)

Confirmed in `src/modern_llm/models/`:

- `attention.py`: MultiHeadAttention with **RoPE**, **GQA**, **attention
  sinks**, SDPA/Flash fallback, head_dim validation.
- `layers.py`: **RMSNorm**, **SwiGLU**.
- `transformer.py`: `DecoderBlock` (pre-norm residual), `ModernDecoderLM`
  with tied embeddings, GPT-style init, causal mask construction.
- `moe.py`: Mixture-of-Experts FFN (unused — see §2.3).

### 2.2 What we must add

| Feature | Status | Why | Where |
|---|---|---|---|
| **QK-norm** | DONE | Scaling Transformers / Chameleon / OLMo-2 — stabilizes attention logits, near-free fix for bf16 NaN spikes | `attention.py`: per-head `RMSNorm(head_dim)` on Q and K before RoPE (order doesn't matter — RMSNorm is invariant to orthogonal rotations). Flag: `ModernLLMConfig.use_qk_norm`, threaded via `DecoderBlock` → `AttentionConfig`. Applies to sink_k on the manual-attention path. Default OFF for back-compat; `configs/lm_75m_2x4090.json` sets it ON. |
| **Residual init scaling** | NEW | Megatron — init residual proj at std = 0.02/sqrt(2*n_layers); critical at 18 layers | `transformer.py` `_init_weights` |
| **Scaled embeddings** | NEW | PaLM, LLaMA — multiply token embeddings by sqrt(d_model) | `transformer.py` forward |
| **Z-loss (1e-4)** | NEW | PaLM §5.1 — penalizes logit norm, stabilizes bf16 without fp32 softmax | training loop |
| **Sequence packing** | NEW | 4k ctx on instruction data is mostly padding without packing | `data/lm_datasets.py` |
| **Streaming dataloader** | DONE | 20B tokens won't fit pre-tokenized in RAM | `data/lm_datasets.py` — already dropped the `NotImplementedError` |
| **`torch.compile`** | NEW | `compile_model: false` today; 4090 + compile is ~20-25% faster | `trainer_base.py` |
| **uP-lite init** | NEW (soft) | Cerebras variant of Tensor Programs V — makes LR transfer more reliable across scaling sweeps | `model_config.py` flag |
| **Attention-sinks OFF for training** | NEW | They break Flash fast-path and are only useful for streaming inference past training context (Xiao et al. 2023 show post-hoc enable works) | config |

### 2.3 What we explicitly **don't** do

- **MoE.** `moe.py` exists but MoE gives params for free, not compute
  for free. Under a 75M hard param cap, experts eat the budget. Skip.
- **Mamba / SSM layers.** Hybrids (Jamba, Zamba) win at 7B+ but show no
  consistent lift at 100M on MMLU/HellaSwag. Skip.
- **Differential Attention (Ye et al. 2024).** Unproven at <1B. Skip.
- **nGPT normalization.** Too new, no public small-model results.
- **ALiBi.** Worse than RoPE on MMLU/long-context. RoPE@theta=500k
  handles 4k easily.

---

## 3. Training-stage roadmap

Five stages. Each has concrete goals, data, HPs, and gating evals.

- **Fast eval** (~1 h on 2x4090): sst2, hellaswag (500), commonsenseqa
  (500), mmlu (1/subject), anli, squad_v2 (200), gsm8k (50).
- **Full eval**: all 16 benchmarks, full sample size.

### Stage 1 — Pretrain

**Goal:** Strong base LM. Clear GPT-2 small on HellaSwag zero-shot (31.1%).

**Pretrain mix (~20B tokens):**

| Dataset | HF name | Tokens | Weight |
|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` (sample-350BT, streamed) | 12B | 60% |
| DCLM-Baseline | `mlfoundations/dclm-baseline-1.0` (10% sample) | 3B | 15% |
| The Stack v2 smol | `bigcode/the-stack-v2-train-smol-ids` (py/js/go) | 2B | 10% |
| OpenWebMath | `open-web-math/open-web-math` | 1B | 5% |
| Wikipedia (en) | `wikimedia/wikipedia`, `20231101.en` | 1B | 5% |
| StackExchange (pile) | `EleutherAI/pile` → stack_exchange | 0.5B | 2.5% |
| arXiv (pile) | `EleutherAI/pile` → arxiv | 0.5B | 2.5% |

**Hyperparameters (DDP on 2x4090, world_size=2):**

| HP | Value |
|---|---|
| Tokens | 20B |
| Seq len | 4096 (packed) |
| Micro-batch per GPU | 4 seqs (16k tok) |
| Grad-accum | 8 |
| Global batch (tokens) | 2 * 4 * 8 * 4096 = **262 144 tok** |
| Steps | ~76 000 |
| Optimizer | AdamW beta=(0.9, 0.95), eps=1e-8 |
| Peak LR | 3e-3 (uP-lite), warmup 2000, cosine to 3e-4 |
| Weight decay | 0.1 (no decay on norms/biases) |
| Grad clip | 1.0 |
| Precision | bf16, fp32 master weights, fp32 norms |
| Dropout | 0.0 |
| Z-loss | 1e-4 |
| Grad checkpointing | On (activations dominate 4090 memory at 4k ctx) |
| Compile | On |
| DDP | `gradient_as_bucket_view=True`, `static_graph=True`, bf16 comm hook |

**Launch:**
```
CUDA_VISIBLE_DEVICES=0,1 ./scripts/launch.sh \
    --config configs/lm_75m_2x4090.json --stage pretrain \
    --nproc-per-node 2
```

**Checkpoints:** rank-0-only via `unwrap_model()` (strips `module.` + `_orig_mod.`);
saved every 2000 steps, keep last 5 + best. All ranks `barrier()` after save.

**Expected:**
- HellaSwag 0-shot acc_norm: 36-40 (GPT-2: 31.1)
- MMLU 5-shot: 27-30 (random=25, GPT-2: 25.9)
- GSM8K: ~0 (unlocked only at Stage 5)

### Stage 2 — Mid-training / anneal

**Goal:** +1-3 MMLU, long-ctx robustness. 1B tokens, high-quality only.

| Dataset | HF name | Tokens | Weight |
|---|---|---|---|
| FineWeb-Edu top decile | `HuggingFaceFW/fineweb-edu` (score>=4) | 500M | 50% |
| Dolmino-mix-1124 | `allenai/dolmino-mix-1124` | 250M | 25% |
| OpenWebMath | `open-web-math/open-web-math` | 150M | 15% |
| TuluMath (CoT prime) | `allenai/tulu-3-sft-mixture` math subset | 50M | 5% |
| Wiki + books | `wikimedia/wikipedia` + Gutenberg subset | 50M | 5% |

- Peak LR = pretrain end-LR (3e-4); warmup 200, cosine to 3e-5.
- Batch/DDP same as pretrain; ~4000 steps.
- Wall-clock: ~6 h on 2x4090.

### Stage 3 — SFT

**Goal:** Instruction following. Unlocks IFEval, IFBench, MixEval-Easy.

| Dataset | HF name | Rows | Weight |
|---|---|---|---|
| Tulu-3 SFT mixture | `allenai/tulu-3-sft-mixture` | 939k | 40% |
| SmolTalk | `HuggingFaceTB/smoltalk` | 1.1M | 20% |
| OpenHermes-2.5 | `teknium/OpenHermes-2.5` | 1M | 15% |
| MetaMathQA | `meta-math/MetaMathQA` | 395k | 10% |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` (200k) | 200k | 5% |
| IFEval-like-data | `argilla/ifeval-like-data` | 56k | 5% |
| NoRobots | `HuggingFaceH4/no_robots` | 10k | 3% |
| CoQA-train | `stanfordnlp/coqa` | 8k | 2% |

- ChatML (`<|im_start|>user / assistant <|im_end|>`); loss masked on prompts.
- Seq 4096 packed; global batch 262k tokens (same shape as pretrain).
- Steps ~4000 (1 epoch).
- Peak LR 5e-5; WD 0.01; dropout 0.1.

### Stage 4 — Preference optimization (DPO)

| Dataset | HF name | Rows |
|---|---|---|
| UltraFeedback (binarized) | `HuggingFaceH4/ultrafeedback_binarized` | 62k |
| HelpSteer2 | `nvidia/HelpSteer2` | 10k |
| SkyworkReward (subset) | `Skywork/Skywork-Reward-Preference-80K-v0.2` | 20k |

- DPO beta=0.05 (small models overfit 0.1).
- Length-normalized DPO (alpha=0.1) to fight DPO length bias.
- Peak LR 3e-6, global batch 32, 1000 steps.
- Reference model = frozen SFT checkpoint (loaded once, eval-mode, no DDP
  wrap on ref).
- Contingency: if MMLU regresses >1 point, WiSE-FT soup
  (0.7 * theta_dpo + 0.3 * theta_sft).

### Stage 5 — Reasoning SFT + optional verifier-guided RL

**Part A — CoT SFT distillation:**

| Dataset | HF name | Rows |
|---|---|---|
| OpenThoughts-114k | `open-thoughts/OpenThoughts-114k` | 114k |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | 860k |
| GSM8K train + self-generated rejection sample | `gsm8k` + RS | 7k + ~20k |

- 2000 steps from DPO checkpoint; LR 1e-5 warmup 100 cosine to 1e-6.

**Part B (conditional on GSM8K < 15% after Part A):**

- Train a small process-reward verifier via `train_verifier.py`.
- Generate 8 samples per GSM8K train item, keep highest-scoring correct
  ones, retrain (STaR / V-STaR).

**Expected end-of-pipeline:** GSM8K 15-25 (GPT-2: ~0), IFEval >20,
MMLU 30-33, HellaSwag >36, MixEval-Easy >30.

---

## 4. Dataset storage policy (re-cache)

### 4.1 Canonical locations

All HF caches live **inside the repo** on the 1.8 TB data volume:

```
data/raw/
  hf_cache/        <- HF_DATASETS_CACHE (arrow shards)
  hf_home/         <- HF_HOME
  hf_home/hub/     <- HF_HUB_CACHE (model weights)
  ... (per-dataset arrow dirs land here automatically)
data/tokenized/    <- our packed uint32 shards
tokenizers/
  cl_small_bpe_16k/   <- custom tokenizer artifact
```

### 4.2 How it's enforced

Three layers:

1. **`scripts/data/migrate_hf_cache.sh`** (idempotent) moves
   `~/.cache/huggingface/{datasets,hub}/*` into `data/raw/{hf_cache,hf_home/hub}/`
   and drops back-symlinks at the original locations so legacy code paths
   still resolve. Safe to re-run.
2. **`scripts/launch.sh`** exports `HF_HOME`, `HF_DATASETS_CACHE`,
   `HF_HUB_CACHE` before execing python.
3. **`src/modern_llm/utils/paths.apply_env_defaults()`** is called at
   import time by `src/modern_llm/data/lm_datasets.py` (and the
   tokenizer-train script). It `setdefault`s the env vars so running
   `python scripts/run_pipeline.py ...` without the launch wrapper still
   redirects the cache. Per-call `cache_dir=cache_dir_for_datasets()` is
   also threaded through `load_dataset()` as belt-and-suspenders.

### 4.3 Disk budget

| Artifact | Size |
|---|---|
| FineWeb-Edu + DCLM (streamed, minor cache) | ~90 GB |
| Other arrow shards (code/math/wiki/stackexchange/arxiv) | ~20 GB |
| Tokenized pretrain uint32 shards | ~80 GB |
| SFT / DPO / reasoning arrow + packed | ~20 GB |
| Model weights hub (Xenova/ada-002 tokenizer legacy + any baselines) | ~5 GB |
| **Total** | **~215 GB** — fits comfortably in 1.8 TB |

---

## 5. Tokenizer decision (overturned)

**Old plan:** keep `Xenova/text-embedding-ada-002` (cl100k, vocab 100 261).
**New plan:** train our own **16k byte-level BPE** on the pretrain mix.

### 5.1 Why train a new one

- cl100k's 100k vocab consumes 51M of the 75M budget at d=512 with tied
  embeddings. That forces `n_layers=8`, which loses to MobileLLM's
  18-layer d=512 depth at the same param count.
- cl100k was designed for OpenAI's English-web mix, not ours. A
  domain-matched tokenizer gives better compression (=> more effective
  tokens per training step) on the actual training distribution.
- 16k vs 32k: we chose **16k** because (a) the extra merges in 32k are
  mostly code/math long-tail tokens that the pretrain mix only needs at
  ~10% weight combined; (b) 16k saves 8.4M params vs 32k, which is
  2-3 additional transformer layers we can spend elsewhere; (c) our seq
  len is 4k — we're not compression-bound on long docs.

### 5.2 Training the tokenizer

```
python3 scripts/data/train_tokenizer.py \
    --vocab-size 16000 \
    --num-samples-per-source 200000 \
    --output-dir tokenizers/cl_small_bpe_16k
```

- Byte-level BPE with NFC normalization and `ByteLevel` pre-tokenizer
  (GPT/LLaMA style). Roundtrip-safe, no OOV.
- Special tokens: `<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`,
  `<|user|>`, `<|assistant|>`, `<|system|>`.
- Wrapped as `PreTrainedTokenizerFast` so
  `AutoTokenizer.from_pretrained("tokenizers/cl_small_bpe_16k")` works.
- `scripts/evaluation/_eval_common.py` `DEFAULT_TOKENIZER` resolves in
  this order: `$MODERN_LLM_TOKENIZER` env var > `tokenizers/cl_small_bpe_16k/`
  if present > `Xenova/text-embedding-ada-002` (legacy fallback for the
  archived pre-4090 checkpoints).

### 5.3 Checkpoint compatibility

- New post-4090 checkpoints use the 16k tokenizer.
- Old pre-4090 checkpoints (under `experiments/runs/gpu-full_archive_pre-4090/`)
  were trained on cl100k. They are **not loadable** into the new
  architecture; the archive exists only for future bisection/re-eval.
- The eval harness's tokenizer fallback still resolves the cl100k
  tokenizer for those archived runs.

---

## 6. Distributed training: GPU-count agnostic DDP

### 6.1 Scope (answer to Q1 / Q2)

- **Single source of DDP truth:** `src/modern_llm/training/distributed.py`.
- Entry points (`train_lm`, `train_sft`, `train_dpo`, `train_verifier`)
  call `init_distributed()` (idempotent), `seed_everything(base_seed)` (each
  rank gets `base_seed + rank`), `get_device()`, `wrap_ddp(model)`. If
  `WORLD_SIZE<=1` `wrap_ddp` returns the bare model — so single-GPU still
  works identically.
- Data: `maybe_distributed_sampler()` attaches a `DistributedSampler` when
  distributed, no-op otherwise. For streaming datasets we use
  `datasets.distributed.split_dataset_by_node()` + per-example tokenization.
- Logging/tqdm gated on `is_main_process()`. `evaluate()` does `all_reduce`
  on loss/batches. Checkpoints saved on rank 0 only, with `barrier()` before
  return. `unwrap_model()` strips both `module.` (DDP) and `_orig_mod.`
  (`torch.compile`) prefixes so a checkpoint saved on 2xGPU loads cleanly
  under 1xGPU or CPU.

### 6.2 Launch layer (answer to Q5 — reconcile NUMA + DDP)

Two modes, both using the same user-facing template:

1. **Single-GPU (user's preferred shape):**
   ```
   CUDA_VISIBLE_DEVICES=1 ./scripts/launch.sh \
       --config gpu --stage all
   ```
2. **Multi-GPU DDP:**
   ```
   CUDA_VISIBLE_DEVICES=0,1 ./scripts/launch.sh \
       --config gpu --stage all --nproc-per-node 2
   ```

`scripts/launch.sh`:

- Exports `HF_HOME` / `HF_DATASETS_CACHE` / `HF_HUB_CACHE` pointed at
  `data/raw/`, creating dirs if missing.
- Sets NCCL defaults for **PCIe consumer GPUs**: `NCCL_P2P_DISABLE=1`
  (RTX 4090 doesn't support P2P over PCIe; without this we get silent
  hangs), `NCCL_IB_DISABLE=1`, `NCCL_ASYNC_ERROR_HANDLING=1`.
- Sets `TOKENIZERS_PARALLELISM=false` (avoids thread storms under
  DataLoader workers).
- NUMA-pins via `numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE`
  (default `NUMA_NODE=1`; override in env). Falls back to no pinning if
  `numactl` is missing.
- Execs `python3 scripts/run_pipeline.py "$@"`.

`scripts/run_pipeline.py`:

- Adds `--nproc-per-node` (default 1).
- If `--nproc-per-node > 1` **and** we are **not** already under torchrun
  (detected via `LOCAL_RANK` being unset), `os.execvp`s
  `torchrun --standalone --nproc_per_node=N <sys.argv minus --nproc-per-node>`.
- Banner prints gated on `RANK==0`.

This is the whole reconciliation: NUMA stays, user's single-GPU shape
stays, multi-GPU is opt-in via one flag.

### 6.3 Archival of pre-4090 runs (Q3)

`scripts/archive_old_runs.sh` moves every `experiments/runs/gpu-full*`
entry into `experiments/runs/gpu-full_archive_pre-4090/`. Idempotent;
re-running after a partial move is a no-op. **Never deletes.**

---

## 7. Compute / time budget

### 7.1 Target hardware

- **Primary: 2x RTX 4090 24 GB, PCIe Gen4, DDP.**
  - bf16; Flash-Attn-2 (Flash-3 requires Hopper).
  - `torch.compile` ON.
  - Grad checkpointing ON (activations for 4k ctx at d=512/18L at bf16
    cost ~9 GB/seq without checkpointing; with checkpointing we fit
    micro-batch 4 per GPU comfortably).
- **Dev: RTX 3060 12 GB** (smoke / config validation).
  Config: `configs/lm_max_rtx3060.json`, seq 1024, micro-batch 2.
- **Optional scale-up:** `--nproc-per-node 4` on a 4x4090 box Just Works.

### 7.2 Throughput estimate

Per-GPU ~75 TFLOP/s sustained bf16 on 4090 (conservative after NCCL +
activation-recompute overhead).

- FLOPs/token ~= 6 * 59e6 ~= 3.5e8.
- 75 TFLOP/s / 3.5e8 ~= **215k tok/s/GPU realistic** (best case without
  checkpointing). With grad checkpointing roughly 0.65x -> ~**140k
  tok/s/GPU** -> ~**280k tok/s** on 2 GPUs (ignoring imperfect DDP
  overlap; budget ~230k tok/s after NCCL overhead).

20B tokens / 230k tok/s = **~24 hours pretrain**.

### 7.3 Wall-clock plan (2x RTX 4090)

| Stage | Tokens / Steps | Time |
|---|---|---|
| Pretrain | 20B | ~24 h |
| Anneal | 1B | ~1.5 h |
| SFT | 0.6B (1 ep) | ~1 h |
| DPO | 40M | ~0.3 h |
| Reasoning SFT | 0.3B | ~0.5 h |
| Verifier + RS (if triggered) | — | ~3 h |
| Evals (full, per stage x 3) | — | ~6 h total |
| **Total** | | **~36 h wall-clock** |

Add a 2x safety margin for restarts, debug, failed runs: **~5 days**.

### 7.4 On RTX 3060 only

Smoke/dev only. A 10k-step pretrain on wikitext-2 with
`configs/lm_max_rtx3060.json` finishes in ~4-6 h; use this to validate
config/eval plumbing before 4090 launches.

---

## 8. Evaluation plan

Unchanged from the previous plan structurally. Targets:

| Benchmark | GPT-2 small (124M) | Our target |
|---|---|---|
| HellaSwag (acc_norm, 10-shot) | 31.1 | **>34.0** |
| MMLU (5-shot) | 25.9 | **>28.0** |
| CommonsenseQA (0-shot) | ~19.5 | **>30.0** |
| ANLI r1 | ~33.1 | **>34.0** |
| BBQ | ~50 | **>52** |
| SQuAD v2 (F1) | ~5-10 | **>25** |
| CoQA (F1) | ~10-15 | **>40** |
| GLUE (avg, 0-shot) | ~35 | **>40** |
| GSM8K (EM, 8-shot CoT) | ~0.5 | **>10** |
| SST-2 (0-shot) | ~50 | **>65** |
| IFEval (strict-instr) | <5 | **>20** |
| IFBench test | <5 | **>10** |
| MixEval-Easy | ~10 | **>30** |
| MMLU-Pro (5-shot) | ~11 | **>14** |
| GPQA (0-shot) | ~25 | **>26** (tie ok) |
| HLE | sub-random | **match or beat random** |

"Beats GPT-2" = **win on >=13 of 16**, including all instruction and
reasoning benchmarks.

---

## 9. Risks and contingencies

| Risk | Probability | Mitigation |
|---|---|---|
| NCCL hang over PCIe on 2x4090 | Medium | `NCCL_P2P_DISABLE=1` set by default in `launch.sh`; `NCCL_ASYNC_ERROR_HANDLING=1` converts hangs to exceptions. |
| Activations OOM at seq=4096 on 24 GB 4090 | Medium | Grad checkpointing ON by default; micro-batch 2 as fallback; seq 2048 if still oom. |
| `torch.compile` breaks inside attention-sinks path | Medium | Sinks OFF for training (§2.2); compile re-evaluated post smoke-run. |
| 20B tokens not enough to beat GPT-2 on MMLU | Medium | Extend pretrain to 30B; or trim architecture to 16 layers and spend the budget on more tokens. |
| DPO regresses MMLU/HellaSwag | Medium | WiSE-FT soup with SFT weights (§3 stage 4). |
| GSM8K doesn't move past 5% | High | Run Stage 5B verifier-guided RL; add OpenMathInstruct-2; self-consistency at eval. |
| Checkpoint saved on 2xGPU fails to load on 1xGPU | Low | `unwrap_model()` strips `module.`/`_orig_mod.` on save; tested at smoke-time. |
| HF dataset download fails / rate-limits | Low | Pile-uncopyrighted + OpenWebText2 fallbacks listed in `download_*.py`. |

---

## 10. Milestones / ordered task list

Phase A tasks in **bold** are already done on this branch; the rest are
upcoming PRs.

### Phase A — Plumbing

1. **M1 DONE**: `HF_HOME` / `HF_DATASETS_CACHE` redirect (`paths.py`,
   `apply_env_defaults()`, `cache_dir_for_datasets()` threaded through
   all loaders; `scripts/data/migrate_hf_cache.sh`).
2. **M2 DONE**: GPU-count-agnostic DDP (`distributed.py`; all 4
   trainers refactored; rank-0 save; `barrier()` sync; DistributedSampler
   / `split_dataset_by_node`; self-spawn `run_pipeline.py`; `launch.sh`
   with NUMA + NCCL PCIe defaults).
3. **M3 DONE**: `scripts/archive_old_runs.sh` (idempotent).
4. **M4 DONE**: `scripts/data/train_tokenizer.py` (16k byte-level BPE).
5. **M5 DONE**: `scripts/evaluation/_eval_common.py` `DEFAULT_TOKENIZER`
   now prefers `tokenizers/cl_small_bpe_16k` when present, falls back to
   cl100k for archived runs.
6. **M6 DONE**: `configs/lm_75m_2x4090.json` with the final 18L/d=512
   architecture, GPU mem 24, world_size 2, bf16, compile ON. `use_qk_norm: true`.
7. **M7 DONE**: QK-norm implemented in `src/modern_llm/models/attention.py`
   (per-head `RMSNorm(head_dim)` on Q and K before RoPE, applied to sink_k
   on the manual path). Flag `use_qk_norm` added to `ModernLLMConfig`,
   `AttentionConfig`, `PipelineConfig`, and threaded through `DecoderBlock`.
   Smoke-tested: works with Flash SDPA, GQA, and attention-sinks paths.
8. **M8 TODO**: Residual init scaling + scaled embeddings + z-loss in
   `transformer.py` / training loop.
9. **M9 TODO**: `torch.compile` toggle in `trainer_base.py` (default
   ON for 4090, OFF for 3060 unless overridden).
10. **M10 TODO**: Turn attention sinks OFF for training in the new
    config (`use_attention_sinks: false`).

### Phase B — Data

11. **M11**: `scripts/data/migrate_hf_cache.sh` invocation + first
    tokenizer train: `python3 scripts/data/train_tokenizer.py` (default
    16k; ~30-45 min on CPU, can run in parallel with anything else).
12. **M12**: `scripts/data/ensure_dataset.py` — symlink-or-download helper.
13. **M13**: `download_pretrain_mix.py` (background; ~6 h streaming).
14. **M14**: `tokenize_pretrain.py` — packed uint32 shards with the new
    16k tokenizer. Output under `data/tokenized/pretrain_mix/`.
15. **M15**: `download_sft_mix.py` + `tokenize_sft.py` (ChatML +
    loss-mask shards).
16. **M16**: `download_dpo_mix.py` + `download_reasoning_mix.py`.

### Phase C — Stage runs

17. **M17**: Pretrain, 2x4090, ~24 h.
18. **M18**: Anneal + fast eval.
19. **M19**: SFT + **full eval** (gate).
20. **M20**: DPO + fast eval + IFEval + MixEval-Easy.
21. **M21**: Reasoning SFT (+ optional 5B verifier RS).
22. **M22**: Full eval on final checkpoint.

### Phase D — Analysis & ship

23. **M23**: `experiments/results/final_report.md` — per-stage deltas
    vs GPT-2 small, GPT-2 medium, SmolLM-2-135M, across all 16 evals.
24. **M24**: Tag final checkpoint at
    `experiments/runs/lm-75m-2x4090/reasoning_final.pt`.

---

## 11. Open questions (resolve before M17)

1. **Grad-checkpointing + `torch.compile`** interaction on 2x4090: verify
   in a 200-step smoke that the compiled graph doesn't recompile per
   step due to dynamic shapes introduced by activation checkpointing.
2. **Confirm NCCL version on the box** — NCCL <2.18 has known deadlocks
   with `NCCL_P2P_DISABLE=1` + grad-accum bucket overlap.
3. **License check** for final model weights: The Stack v2 smol is the
   usual gotcha; confirm acceptance before publishing.
4. **Do we ship the 16k tokenizer alongside the model?** Yes, because
   evals depend on it (it's not a HF Hub tokenizer).

---

**End of plan.**
