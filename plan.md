# Plan: Beating GPT-2 (124M) With a Sub-75M Modern LM

**Author:** ML Research
**Date:** 2026-04-17
**Repo:** `/workspace/mnt/data_sda/lost+found/nikhil/modern_llm`
**Target:** A from-scratch decoder-only LM with < 75M parameters that strictly
outperforms GPT-2 small (124M) on the 16-benchmark suite listed in
`scripts/evaluation/` (gpqa, hle, ifbench_test, ifeval, mixeval_easy, mmlu_pro,
anli, bbq, commonsenseqa, coqa, glue, hellaswag, mmlu, squad_v2, gsm8k, sst2).

The current SFT checkpoint at `experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt`
scores 49.3% SST-2 and 0.0% GSM8K. That tells us the existing pipeline has
the right skeleton but is catastrophically underfit on data (wikitext-2, 20k
steps) and has no reasoning data in its diet. The plan below is designed to
close that gap with a carefully sized architecture, a Chinchilla-compliant
data budget, and a modern 5-stage recipe.

---

## 0. Executive summary

- **Architecture:** 14-layer decoder-only Transformer, `d_model=576`,
  `n_heads=9` (head_dim=64), `n_kv_heads=3` (GQA 3:1), SwiGLU FFN with
  `d_ff=1536` (≈8/3·d), RoPE (theta=500k), RMSNorm pre-norm with QK-norm,
  untied embeddings, sequence length 4096. **Total params: ~74.2M**
  (non-embedding ~53.0M).
- **Tokenizer:** keep `Xenova/text-embedding-ada-002` (cl100k, vocab 100261).
  Already used by existing checkpoints and all 16 eval scripts. Switching is
  not worth the cost right now.
- **Token budget:** ~20B pretrain + 1B anneal + 0.6B SFT + 0.1B preference,
  total ≈ **21.7B training tokens**. This is ≈290 tokens/param for
  pretrain — roughly 15× over Chinchilla-optimal and deliberately so, per
  SmolLM-2 / MobileLLM which show sub-100M models benefit massively from
  token overtraining.
- **Stages:** (1) pretrain, (2) mid-training / anneal, (3) SFT, (4) DPO,
  (5) reasoning distillation + RL with verifier. Gated by evals between
  stages.
- **Dataset storage:** all datasets land under `data/raw/` inside this repo,
  with `HF_DATASETS_CACHE` / `HF_HOME` redirected there. Nothing goes into
  `~/.cache/huggingface` by default.
- **Target hardware:** 1× H100 (80 GB) for the heavy lifting, RTX 3060
  (12 GB) for dev and smoke tests. Wall-clock ≈ 7 days on H100.

### The 5 big bets (why we think this beats GPT-2)

1. **Data quality, not model size.** GPT-2 was trained on 40 GB of
   WebText; we train on ~20B tokens of FineWeb-Edu + DCLM-Baseline + code +
   math. SmolLM-2 360M beats GPT-2 XL on MMLU with this recipe; we reuse the
   recipe in a smaller envelope.
2. **Depth over width.** MobileLLM (Liu et al., 2024) shows that for
   <1B LMs, **deeper & thinner** (high `n_layers`, modest `d_model`) beats
   the square aspect ratio. We go 14 layers at `d_model=576` vs. GPT-2
   small's 12×768.
3. **Modern stack (RoPE + GQA + SwiGLU + QK-norm + RMSNorm).** Each of
   these is individually small; compounded they're worth ~1-2 nats/token at
   this scale and give us long-context evals (MMLU-Pro, CoQA) essentially
   for free.
4. **Instruction + preference + reasoning distillation.** GPT-2 is a raw
   LM; our final model sees SFT on Tulu-3 + SmolTalk, DPO on UltraFeedback,
   then reasoning distillation from OpenThoughts / NuminaMath. This alone
   dominates on IFEval, IFBench, GSM8K, and MixEval-Easy, where base GPT-2
   simply cannot follow instructions.
5. **Mid-training anneal on high-quality + long context.** A 1B-token
   annealing phase (Dolmino / FineWeb-Edu top decile / OpenWebMath) at
   10× lower LR lifts MMLU by 2-5 pts at this scale without changing params.

---

## 1. Parameter budget

### 1.1 Architectural sizing (final numbers)

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_layers` | 14 | MobileLLM depth-over-width finding for <1B models |
| `d_model` | 576 | Divisible by 9 and 64; head_dim=64 is flash-attn sweet spot |
| `n_heads` (Q) | 9 | `d_model / head_dim = 576 / 64 = 9` |
| `n_kv_heads` | 3 | GQA with 3:1 ratio; ~33% KV cache, minor quality hit |
| `head_dim` | 64 | Flash-Attn-3 optimal; keeps `d_model` divisible |
| `d_ff` | 1536 | ≈ 8/3 · d_model, rounded to 64× (LLaMA/SwiGLU convention) |
| `vocab_size` | 100 261 | Keep existing tokenizer to preserve evals |
| `max_seq_len` | 4096 | RoPE base 500 000 supports this |
| `tie_embeddings` | **False** | Untied helps perplexity ~0.1 at this scale; we can afford 57M extra params only if we stay under 75M cap — see math below |
| `rope_theta` | 500 000 | LLaMA-3 convention; extrapolates cleanly to 8k |
| `dropout` | 0.0 pretrain / 0.1 SFT | Modern practice: no dropout during pretrain |
| `init` | std=0.02, scaled 1/√(2·n_layers) for residual projections | GPT-2 init + residual scaling (Megatron) |

### 1.2 Parameter-count table

Non-embedding parameters per block (using `d=576`, `d_ff=1536`, `n_heads=9`, `n_kv=3`, `head_dim=64`):

| Component | Formula | Count |
|---|---|---|
| Q projection | `d · d = 576·576` | 331 776 |
| K projection | `d · (n_kv·head_dim) = 576·192` | 110 592 |
| V projection | `d · (n_kv·head_dim) = 576·192` | 110 592 |
| O projection | `d · d = 576·576` | 331 776 |
| SwiGLU (3 mats) | `3 · d · d_ff = 3·576·1536` | 2 654 208 |
| RMSNorm ×2 | `2 · d` | 1 152 |
| QK-norm ×2 | `2 · head_dim` | 128 |
| **Per block** | | **3 540 224** |

Totals:

| Bucket | Count |
|---|---|
| 14 × decoder block | 49 563 136 |
| Final RMSNorm | 576 |
| Token embedding (`vocab · d = 100261·576`) | 57 750 336 |
| LM head (untied, `d · vocab`) | 57 750 336 |
| Block-internal params | 49 563 136 |
| **Total (untied)** | **165 064 384** |

That's **165M**, way over budget. So **we must tie embeddings**. With
`tie_embeddings=True`:

| Bucket | Count |
|---|---|
| Token embedding = LM head (shared) | 57 750 336 |
| 14 × decoder block | 49 563 136 |
| Final RMSNorm | 576 |
| **Total (tied)** | **107 314 048** |

Still over 75M. The embedding is 78% of a tied model's params because of
the 100k vocab. Two realistic knobs:

- **Option A — trim vocab.** Switch to a 49 152-token tokenizer (SmolLM-2's
  choice) and keep the current architecture: `49152·576 = 28 311 552` embed
  + 49 563 136 block = **77.9M**, still just over. Forces a retokenization
  of all evals.
- **Option B — shrink `d_model` to 512.** Recompute with `d=512`, `d_ff=1408`,
  `head_dim=64`, `n_heads=8`, `n_kv=2`:
  - Block: Q/O = 2·(512·512) = 524 288; K/V = 2·(512·128) = 131 072;
    SwiGLU = 3·512·1408 = 2 162 688; norms ≈ 1 152.
    Per block: **2 819 200**.
  - 14 blocks: 39 468 800.
  - Tied embed: `100261·512 = 51 333 632`.
  - **Total: 90.8M.** Still over.
- **Option C (chosen) — `d_model=512`, `n_layers=10`, `d_ff=1408`,
  tied embeddings, `head_dim=64`:**
  - 10 × 2 819 200 = 28 192 000 block params
  - Tied embed: 51 333 632
  - Norms: 512
  - **Total: 79.5M.** Still 4.5M over.
- **Option D (final) — `d_model=512`, `n_layers=8`, `d_ff=1408`, tied:**
  - 8 × 2 819 200 = 22 553 600
  - Tied embed: 51 333 632
  - **Total: 73.9M.** ✅ Under 75M.

With the cl100k tokenizer the embedding cost dominates and we cannot afford
a 14-layer model. The **final choice is Option D**:

| Hyperparameter | Final Value |
|---|---|
| `n_layers` | **8** |
| `d_model` | **512** |
| `n_heads` | **8** (head_dim=64) |
| `n_kv_heads` | **2** (GQA 4:1) |
| `d_ff` | **1408** (rounded from 8/3·512 to nearest 64) |
| `vocab_size` | **100 261** (cl100k, keep) |
| `max_seq_len` | **4096** |
| `tie_embeddings` | **True** |
| **Total params** | **≈73.9M** |
| **Non-embedding params** | **≈22.6M** |

### 1.3 Contrast with GPT-2 small (124M)

| Component | GPT-2 small | Our model | Savings |
|---|---|---|---|
| `n_layers` | 12 | 8 | −33% |
| `d_model` | 768 | 512 | −33% |
| `d_ff` | 3072 | 1408 | −54% |
| Attention | MHA, 12 heads, learned pos | GQA 8/2, RoPE | KV cache 4× smaller |
| Norm | LayerNorm (post-norm-ish) | RMSNorm + QK-norm | ~1-2% speed, better stability |
| FFN | GELU(Wx)W' | SwiGLU (gated) | Better sample efficiency |
| Embedding | 50 257 · 768 = 38.6M | 100 261 · 512 = 51.3M (tied) | +33% (tokenizer is bigger) |
| Non-embed | ~85M | ~22.6M | **−73%** |
| Total | 124M | 73.9M | −40% |

**The savings come from the transformer blocks, not the embeddings.** Our
embedding is actually *larger* than GPT-2's because we use a 2× bigger
vocab. We absorb that cost and still beat GPT-2 because the blocks are
dramatically more parameter-efficient (deeper-than-GPT-2 effective compute
per param with SwiGLU/GQA/RoPE/QK-norm).

### 1.4 Caveat on "depth over width"

Option D went narrower and shallower than I wanted because of the big
tokenizer. If we commit to a retokenization (Option A), a 14-layer /
576-dim model becomes feasible and should give ≈1 pt more on MMLU by
MobileLLM's scaling curves. **Recommendation: ship Option D first; if
MMLU stalls below GPT-2 after full training, execute Option A (tokenizer
swap) as the primary lever.**

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

Verdict: the scaffold is production-quality. No rewrites needed. We add
features on top.

### 2.2 What we must add

| Feature | Status | Why | Where |
|---|---|---|---|
| **QK-norm** | NEW | Scaling Transformers (Henry et al. 2020; Chameleon 2024) — stabilizes attention logits, near-free and fixes occasional NaN spikes we've seen in bf16 | `attention.py`: add RMSNorm on Q and K before RoPE |
| **Residual init scaling** | NEW | GPT-2 paper §2.3, Megatron — init residual proj at std = 0.02/√(2·n_layers) — stops deep stacks from blowing up | `transformer.py` `_init_weights` |
| **Scaled embeddings** | NEW | PaLM, LLaMA — multiply token embeddings by √d_model; tiny but measurable | `transformer.py` forward |
| **Z-loss (1e-4)** | NEW | PaLM §5.1 — penalizes logit norm, stabilizes bf16 training without needing fp32 softmax | training loop |
| **Sequence packing** | NEW | 4k contexts at 73M params are mostly padding on wiki/openorca — packed sequences with attention-mask resets give 2-3× throughput | `data/lm_datasets.py` |
| **Streaming dataloader** | NEW | 20B tokens cannot fit in RAM pre-tokenized; must stream | `data/lm_datasets.py` (drop the `streaming=False` hard-coded exception) |
| **`torch.compile`** | NEW | `compile_model: false` in current config. H100 + compile is ~25% faster; needs an `is_causal=True` SDPA path that's already there | `trainer_base.py` |
| **µP-lite** | NEW (soft) | Tensor Programs V (Yang 2022) — full µP is overkill, but apply the key rule: attention logits scale is `1/d_k` not `1/√d_k` only for the *final* layer, and init scales are `1/√d`. We'll use the simpler "Cerebras µP" variant | `model_config.py` flag |
| **Sliding-window + full-attention hybrid** | OPTIONAL (defer) | Mistral-style; useful for long context but we cap at 4k — defer to v2 | — |

### 2.3 What we explicitly **don't** do

- **MoE.** The `moe.py` module is present but MoE gives you *parameters
  for free*, not *compute for free*. At a 75M hard cap, MoE experts eat the
  budget. Only useful if we relax to "active params < 75M". Skip.
- **Mamba / SSM layers.** Hybrid Mamba-Transformers (Jamba, Zamba) win
  at 7B+ but show no consistent lift at 100M on the evals we care about
  (MMLU, HellaSwag). Adds engineering complexity. Skip.
- **Differential Attention (Ye et al. 2024).** Interesting but unproven at
  <1B. Adds 2× Q/K projections (eating our budget). Skip.
- **nGPT normalization.** Too new, no public small-model results on our
  evals. Skip.
- **ALiBi.** Worse than RoPE on MMLU/long-context (Press 2022 original,
  refuted by LLaMA); RoPE+theta=500k already handles 4k easily. Skip.

### 2.4 Attention sinks — keep or drop?

The repo has attention sinks on by default (`use_attention_sinks: true`,
`num_attention_sinks: 4`). They (1) break Flash Attention (forcing the
slower manual path) and (2) are only useful for streaming inference
beyond the training context, which we don't do. **Turn them off for
training.** They can optionally be re-enabled at inference time for
long-context evals (Xiao et al. 2023 show this works post-hoc).

---

## 3. Training-stage roadmap

Five stages. Each has concrete goals, data, HPs, and gating evals. The
eval set is split into **fast** (cheap, runs every stage end) and **full**
(expensive, runs only after stages 3-5).

- **Fast eval** (≈1 hr on 1×H100): sst2, hellaswag (500 samples),
  commonsenseqa (500), mmlu (1 sample/subject), anli, squad_v2 (200),
  gsm8k (50).
- **Full eval**: all 16 benchmarks at full sample size.

### Stage 1 — Pretrain

**Goal:** Strong base LM. Target: ≥ GPT-2 small on HellaSwag zero-shot
(GPT-2-small: 31.1% acc_norm).

**Datasets (pretrain mix, ≈20B tokens):**

| Dataset | HF name | Tokens (approx) | Weight | License | Why |
|---|---|---|---|---|---|
| FineWeb-Edu (350BT sample) | `HuggingFaceFW/fineweb-edu` (sample-350BT) | 12B | 60% | ODC-By | Highest-quality web text; primary driver of MMLU at this scale |
| DCLM-Baseline | `mlfoundations/dclm-baseline-1.0` (10% sample) | 3B | 15% | CC-BY-4.0 | Complements FineWeb-Edu — different filter, similar quality |
| The Stack v2 smol | `bigcode/the-stack-v2-train-smol-ids` (filtered to Python/JS/Go) | 2B | 10% | permissive subset | Code. GSM8K and reasoning benchmarks correlate strongly with code exposure |
| OpenWebMath | `open-web-math/open-web-math` | 1B | 5% | ODC-By | Math web text — cheap GSM8K lift |
| StackExchange (pile subset) | `EleutherAI/pile` → stack_exchange split | 0.5B | 2.5% | CC-BY-SA | Q&A structure; helps CommonsenseQA / MMLU-Pro |
| Wikipedia (en) | `wikimedia/wikipedia`, `20231101.en` | 1B | 5% | CC-BY-SA | Factual grounding for MMLU/GPQA |
| arXiv (pile subset) | `EleutherAI/pile` → arxiv | 0.5B | 2.5% | arXiv license | Scientific text for GPQA/MMLU-Pro |

**Hyperparameters:**

| HP | Value |
|---|---|
| Tokens | 20B |
| Seq len | 4096 (packed) |
| Global batch (tokens) | 524 288 (128 seqs × 4096) |
| Steps | ~38 000 |
| Optimizer | AdamW β=(0.9, 0.95), eps=1e-8 |
| Peak LR | 3e-3 (µP-lite), warmup 2 000 steps, cosine decay to 3e-4 |
| Weight decay | 0.1 (no decay on norms/biases) |
| Grad clip | 1.0 |
| Precision | bf16, fp32 master weights, fp32 norms |
| Dropout | 0.0 |
| Z-loss coef | 1e-4 |
| Gradient checkpointing | Off on H100, on for RTX 3060 |

**Checkpoints:** save every 2 000 steps, keep last 5 + best.

**End-of-stage evals:** fast eval set + pretrain ppl on FineWeb-Edu val.

**Expected:**
- HellaSwag (zero-shot, acc_norm): 34–38% (GPT-2: 31.1%)
- MMLU (5-shot): 26–29% (random = 25%, GPT-2: 25.9%)
- ARC-easy (proxy): 45–50%
- GSM8K: still ~0%, reasoning unlocked only in stage 5.

### Stage 2 — Mid-training / anneal

**Goal:** +1–3 MMLU points and long-context robustness. Extend effective
context to 4k (already trained at 4k, but anneal sees more long docs).

**Datasets (1B tokens, high-quality only):**

| Dataset | HF name | Tokens | Weight | Why |
|---|---|---|---|---|
| FineWeb-Edu top decile | `HuggingFaceFW/fineweb-edu` (score≥4) | 500M | 50% | Textbook-quality web |
| Dolmino-mix-1124 | `allenai/dolmino-mix-1124` | 250M | 25% | Curated mid-training mix from OLMo 2 |
| OpenWebMath | `open-web-math/open-web-math` | 150M | 15% | Math lift |
| TuluMath | `allenai/tulu-3-sft-mixture` math subset | 50M | 5% | Chain-of-thought priming |
| Wiki + books | `wikimedia/wikipedia` + Project Gutenberg subset | 50M | 5% | Clean long docs |

**Hyperparameters:**

- Peak LR = pretrain end-LR (3e-4), linear warmup 200 steps, cosine decay
  to 3e-5.
- Batch size same; ~2 000 steps.
- Duration: ≈ 6h on H100.

**Checkpoint:** `anneal_final.pt`.

**Evals:** fast set.

**Expected lift:** +2 MMLU, +1 HellaSwag, +0.5 ARC.

### Stage 3 — SFT

**Goal:** Instruction following. Unlock IFEval, IFBench, MixEval-Easy,
MMLU-Pro (zero-shot w/ instruction).

**Datasets (mix ~0.6B tokens after formatting):**

| Dataset | HF name | Rows | Weight | Why |
|---|---|---|---|---|
| Tulu-3 SFT mixture | `allenai/tulu-3-sft-mixture` | 939k | 40% | Best single OSS SFT mix; spans chat, math, code, safety |
| SmolTalk | `HuggingFaceTB/smoltalk` | 1.1M | 20% | SmolLM-2's SFT mix; short, diverse |
| OpenHermes-2.5 | `teknium/OpenHermes-2.5` | 1M | 15% | High-quality synthetic instructions |
| MetaMathQA | `meta-math/MetaMathQA` | 395k | 10% | GSM8K-aligned math CoT |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` (sample 200k) | 200k | 5% | Harder math with NeMo-Skills CoT |
| NoRobots | `HuggingFaceH4/no_robots` | 10k | 3% | Human-written, quality floor |
| IFEval-like-data | `argilla/ifeval-like-data` | 56k | 5% | Direct prep for IFEval/IFBench constraints |
| CoQA-train | `stanfordnlp/coqa` | 8k | 2% | Direct prep for CoQA eval |

**Formatting:** ChatML with `<|im_start|>user / assistant <|im_end|>`.
Loss masked on prompt tokens.

**Hyperparameters:**

- Seq len 4096, packed.
- Global batch 64 (≈ 262k tokens).
- Steps: 4 000 (≈ 1 epoch over mix).
- Peak LR 5e-5, warmup 100, cosine to 5e-6.
- Weight decay 0.01. Dropout 0.1 (to regularize against synthetic data
  style).
- Epochs: 1 (SmolLM-2 showed >1 epoch hurts on Tulu-mix).

**Checkpoint:** `sft_final.pt`.

**Evals:** **Full eval set.**

**Expected numbers (targets, not guarantees):**

- IFEval strict-instr: 20–30% (GPT-2 instruction-tuned baselines: <5%).
- IFBench: 10–15%.
- MixEval-Easy: 30–40%.
- GSM8K (greedy CoT): 8–15%.
- MMLU (5-shot): 30–33%.
- HellaSwag: +0.5–1 over anneal.

### Stage 4 — Preference optimization (DPO)

**Goal:** Response quality, safety, chat UX. Small but non-zero lift on
MixEval-Easy and MMLU-Pro (zero-shot).

**Datasets:**

| Dataset | HF name | Rows | Why |
|---|---|---|---|
| UltraFeedback (binarized) | `HuggingFaceH4/ultrafeedback_binarized` | 62k | Gold standard for DPO |
| HelpSteer2 | `nvidia/HelpSteer2` | 10k | Dense, trusted reward signal |
| SkyworkReward | `Skywork/Skywork-Reward-Preference-80K-v0.2` (subset 20k) | 20k | Additional preference diversity |

**Hyperparameters:**

- DPO β = 0.05 (lower than usual; small models overfit β=0.1).
- Peak LR 3e-6, warmup 50, cosine to 3e-7.
- Global batch 32, 1 000 steps.
- RPO / length-normalized variant: length-normalized DPO (`α=0.1`) to
  prevent the known DPO-length-bias.
- Reference model: frozen SFT checkpoint.

**Checkpoint:** `dpo_final.pt`.

**Evals:** fast set + IFEval + MixEval-Easy.

**Expected:** +2–4 IFEval, +1–2 MixEval-Easy, no regression elsewhere.
**Contingency:** if MMLU regresses >1pt, blend DPO and SFT weights
(`θ_final = 0.7·θ_dpo + 0.3·θ_sft`, "WiSE-FT"-style model soup).

### Stage 5 — Reasoning distillation + verifier-guided RL

**Goal:** GSM8K. This is where we beat GPT-2 most decisively.

**Part A — SFT-on-CoT distillation:**

| Dataset | HF name | Rows | Why |
|---|---|---|---|
| OpenThoughts-114k | `open-thoughts/OpenThoughts-114k` | 114k | Distilled R1-style long CoTs |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | 860k | Competition math CoTs |
| GSM8K train + self-generated | `gsm8k` + rejection-sampled | 7k + ~20k | Task-aligned CoT |

- Train for 2 000 steps on this mix starting from the DPO checkpoint.
- LR 1e-5, warmup 100, cosine to 1e-6.
- Resulting checkpoint: `reasoning_sft.pt`.

**Part B — Verifier-guided RL (optional, only if Part A < 15% GSM8K):**

- Use the existing `train_verifier.py` pipeline to train a small
  process-reward model on GSM8K labeled reasoning.
- Run rejection sampling: generate 8 samples per GSM8K train question,
  keep ones the verifier scores highest and correct, retrain with this as
  expert iteration (STaR / V-STaR).
- Budget: 2 more SFT rounds.

**Checkpoint:** `reasoning_final.pt`.

**Evals:** **Full eval set.** Primary targets: GSM8K, MMLU-Pro, GPQA.

**Expected:** GSM8K 15–25% (GPT-2: ~0%), GPQA ~25% (random=25%; GPT-2
is sub-random here).

### Stage 6 — Final model & evaluation gate

- Run `evaluate_tasks.py` on {pretrain, anneal, sft, dpo, reasoning_final}
  against all 16 benchmarks.
- Update `experiments/results/stage_gains.md` and
  `experiments/results/baseline_comparison.md` with per-stage deltas.
- Ship `reasoning_final.pt` as the headline model.

---

## 4. Dataset storage policy

### 4.1 Target directory structure

All dataset artifacts live **inside this repo**, under
`/workspace/mnt/data_sda/lost+found/nikhil/modern_llm/data/`:

```
data/
  raw/                      # HF_DATASETS_CACHE target (arrow shards from load_dataset)
    fineweb-edu/
    dclm-baseline/
    the-stack-v2-smol/
    open-web-math/
    dolmino-mix-1124/
    tulu-3/
    smoltalk/
    openhermes-2.5/
    metamath-qa/
    openmathinstruct-2/
    ultrafeedback/
    helpsteer2/
    skywork-reward/
    openthoughts-114k/
    numinamath-cot/
    wikipedia-en/
    pile-subsets/
  tokenized/                # our pre-packed .bin/.npy shards produced by scripts/data/tokenize_*.py
    pretrain_mix/
    anneal_mix/
    sft_mix/
    dpo_mix/
    reasoning_mix/
  eval/                     # eval-time datasets (also cached via HF_DATASETS_CACHE=data/raw)
```

### 4.2 Enforcing the cache redirect

Every download and every `load_dataset()` call must route through
`data/raw/`. Canonical snippet that all scripts (`scripts/data/*.py`,
`scripts/evaluation/*.py`, `src/modern_llm/data/*.py`) should use:

```python
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # or [1], adjust per file depth
DATA_ROOT = REPO / "data" / "raw"
os.environ["HF_HOME"] = str(REPO / "data" / "hf_home")
os.environ["HF_DATASETS_CACHE"] = str(DATA_ROOT)
os.environ["HF_HUB_CACHE"] = str(REPO / "data" / "hf_home" / "hub")

from datasets import load_dataset
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-350BT",
    split="train",
    streaming=True,
    cache_dir=str(DATA_ROOT),
)
```

Additionally:

- Check for **existing** presence in the default HF cache first
  (`~/.cache/huggingface/datasets/<name>`); if found, symlink into
  `data/raw/` instead of re-downloading. A helper
  `scripts/data/ensure_dataset.py` should wrap this logic.
- For tokenizers (`AutoTokenizer.from_pretrained`): set
  `cache_dir=REPO/"data"/"hf_home"/"hub"` or rely on `HF_HOME`.
- Commit a top-level `.env.example` with the three exports so anyone
  running the pipeline gets the same behavior:
  ```
  HF_HOME=./data/hf_home
  HF_DATASETS_CACHE=./data/raw
  HF_HUB_CACHE=./data/hf_home/hub
  ```
- Update `pyproject.toml` / `scripts/run_pipeline.py` entrypoint to
  `source .env` or `dotenv`-load at startup.

### 4.3 Datasets already in the HF cache

I did **not** run a full scan — but every download script written for
this plan must first check `~/.cache/huggingface/datasets/` and
`~/.cache/huggingface/hub/` for a given dataset and, if present,
symlink (not copy) into `data/raw/`. Only download fresh if absent. This
avoids blowing up the 1.8 TB budget on `sda` with duplicates.

### 4.4 Disk budget

Approximate raw (arrow) + tokenized (uint32, 4 bytes/token) usage:

| Mix | Raw text (GB) | Tokenized (GB) |
|---|---|---|
| Pretrain (20B tokens) | ~90 GB (streamed, minor cache) | 80 |
| Anneal (1B) | ~8 GB | 4 |
| SFT (~2M rows, 0.6B tokens after packing) | ~5 GB | 2.4 |
| DPO (~90k rows) | 1 GB | 0.4 |
| Reasoning (1M rows) | 6 GB | 3 |
| Raw HF shards (full FineWeb-Edu sample-350BT if we pull full) | up to 1.2 TB |  |

**Budget:** cap FineWeb-Edu + DCLM to **streaming + on-the-fly caching**;
do **not** materialize the full 1.2 TB. Realistic footprint:
**≈180 GB** arrow shards + **≈100 GB** tokenized binaries = **≈280 GB**.
That fits comfortably in the 1.8 TB available on `sda`.

---

## 5. Tokenizer decision

**Keep `Xenova/text-embedding-ada-002` (cl100k_base, vocab 100 261).**

Why:

- All 16 eval scripts under `scripts/evaluation/` assume this tokenizer
  (`DEFAULT_TOKENIZER` in `_eval_common.py`).
- The existing checkpoint at
  `experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt` uses
  this tokenizer. Switching invalidates it and forces a cold start on
  fast iteration.
- cl100k has excellent compression on code and math (better than GPT-2
  BPE), which matters for our data mix.

**Cost:** the 100k vocab eats 51M of our 74M budget (69%). This is the
binding constraint; it is why we ended up at 8 layers.

**Conditional switch (only if Option D plateaus):** swap to
`HuggingFaceTB/cosmo2-tokenizer` (vocab 49 152) at the cost of
retokenizing evals and retraining. This frees ~25M params — enough for
a 14-layer, d=576 model that matches SmolLM-2's architecture 1:1.

---

## 6. Compute / time budget

### 6.1 Target hardware

- **Primary: 1× H100 80GB** (bf16, Flash-Attn-3, `torch.compile`).
- **Dev: RTX 3060 12GB** (for smoke tests, config validation). Gradient
  checkpointing on, micro-batch 2, seq 1024. `configs/lm_max_rtx3060.json`
  already targets this.
- **Optional: 2-8× H100** — the training loop should be DDP-safe; the
  existing `HardwareConfig.from_env()` already reads `LOCAL_RANK`.

### 6.2 Throughput estimates (1× H100, bf16, compile, flash-attn)

Back-of-envelope with 6·N (forward + backward = ~6× fwd flops) for a
74M model:

- FLOPs/token ≈ 6 · 74e6 ≈ 4.4e8.
- H100 sustained ~400 TFLOP/s bf16 ≈ 2e14 FLOP/s usable → **~450k
  tok/s** theoretical, **~150k tok/s** realistic after IO + sync.
- 20B tokens / 150k tok/s = **~37 hours** for pretrain.

### 6.3 Wall-clock plan (1× H100)

| Stage | Tokens / Steps | Time |
|---|---|---|
| Pretrain | 20B tokens | 40 h |
| Anneal | 1B | 2 h |
| SFT | 0.6B (1 ep) | 2 h |
| DPO | 40M | 0.5 h |
| Reasoning SFT | 0.3B | 1 h |
| Verifier + RS | — | 4 h |
| Evals (full, per stage × 3) | — | 6 h total |
| **Total** | | **≈ 56 h (≈ 2.5 days)** |

Add a 2× safety margin for restarts, bugs, failed runs: **7 days wall-clock**.

### 6.4 On RTX 3060 only

Not feasible for the full 20B pretrain (would take ~2 months). Use it for:
- smoke tests (`configs/lm_max_rtx3060.json` → 20k steps, wikitext).
- SFT/DPO of a pretrained checkpoint (fits if grad checkpoint + micro_bs=1).
- Inference / eval runs.

---

## 7. Evaluation plan

### 7.1 Cadence

| Stage end | Evals run | Wall-clock |
|---|---|---|
| Pretrain | fast set | ~1 h |
| Anneal | fast set | ~1 h |
| SFT | full set | ~4 h |
| DPO | fast + ifeval + mixeval_easy | ~1.5 h |
| Reasoning | full set | ~4 h |

All evals write JSON under `experiments/results/<run>/<stage>/<task>.json`.
A new `scripts/evaluation/update_dashboards.py` aggregates them into:

- `experiments/results/baseline_comparison.md` (our best vs. GPT-2,
  GPT-2-medium, SmolLM-2-135M).
- `experiments/results/stage_gains.md` (Δ per stage, like it does today
  but extended to 16 benchmarks).

### 7.2 Published GPT-2 (124M) numbers to beat

Pulled from the official GPT-2 paper, HELM, and lm-eval-harness community
results. Where we use harness defaults (5-shot MMLU, 10-shot HellaSwag,
zero-shot ANLI, etc.), these are the numbers we must clear:

| Benchmark | GPT-2 small (124M) | Our target |
|---|---|---|
| HellaSwag (acc_norm, 10-shot) | 31.1 | **>34.0** |
| MMLU (5-shot acc) | 25.9 (random=25) | **>28.0** |
| ARC-E / CommonsenseQA (acc, 0-shot) | ~19.5 | **>30.0** |
| ANLI r1 (acc) | ~33.1 (random=33.3) | **>34.0** |
| BBQ (acc, ambiguous neg-bias-aware) | ~50 | **>52** |
| SQuAD v2 (F1, 1-shot) | ~5-10 (no QA training) | **>25** |
| CoQA (F1) | ~10-15 | **>40** |
| GLUE (avg) | finetuned: 78 / zero-shot: ~35 | **>40 zero-shot** |
| GSM8K (EM, 8-shot CoT) | ~0.5 | **>10** |
| SST-2 (0-shot) | ~50 (random) | **>65** |
| IFEval (strict-instr) | <5 (not instruct-tuned) | **>20** |
| IFBench test | <5 | **>10** |
| MixEval-Easy | ~10 | **>30** |
| MMLU-Pro (5-shot) | ~11 (random=10) | **>14** |
| GPQA (0-shot) | ~25 (random=25) | **>26** (noisy — tie OK) |
| HLE | sub-random | **match or beat random** |

"Beating GPT-2" = **win on at least 13 of 16** benchmarks, including all
instruction-following and reasoning benchmarks.

### 7.3 Regression tracking

- After each stage, diff the new row in `task_metrics.json` against the
  previous stage's. Any task that regresses > 2 points flags an alert.
- If DPO regresses MMLU, HellaSwag, or GSM8K by >2 points → trigger the
  model-soup contingency (§3, Stage 4).
- If Reasoning-stage regresses IFEval → reduce reasoning-mix weight and
  blend reasoning-SFT with DPO checkpoint (60/40).

---

## 8. Risks and contingencies

| Risk | Probability | Mitigation |
|---|---|---|
| Pretrain loss plateaus / NaN in bf16 | Medium | QK-norm (§2.2) + z-loss + fp32 norms; restart from last good checkpoint at 0.5× LR |
| 20B tokens not enough to beat GPT-2 on MMLU | Medium | Extend pretrain to 30B, or execute **Option A** (tokenizer swap → 14-layer model) |
| 4k context too long for packed SFT data | Low | Fall back to 2k; IFEval/CoQA prompts are <1k |
| DPO hurts instruction following | Medium | WiSE-FT model soup with SFT weights; or switch to SimPO (reference-free) |
| GSM8K doesn't move past 5% | High | (a) Run Stage 5B verifier-guided RL; (b) add more OpenMathInstruct-2; (c) increase CoT temperature at eval + majority vote (self-consistency, 8 samples) |
| Tokenizer vocab too large blows param budget | Already happened | Option D (n_layers=8) accepted; conditional Option A if stuck |
| H100 unavailable | Medium | Plan still works on 4× A100-40GB with FSDP; already supported via `torchrun`. Wall-clock doubles |
| Dataset download fails (licensing / rate limit) | Low | Fall back dataset list: Pile-uncopyrighted, OpenWebText-2 for web; skip FineWeb-Edu if blocked |
| Eval scripts broken on new checkpoints | Low | `_eval_common.py` loader is already robust; smoke-test after every checkpoint save |

---

## 9. Milestones / ordered task list

This is the execution plan. Each milestone (Mx) is a concrete PR.

### Phase A — Plumbing (1–2 days of eng work)

1. **M1**: Add `HF_HOME` / `HF_DATASETS_CACHE` redirect to `data/raw/`.
   - Edit `src/modern_llm/data/lm_datasets.py` and every
     `scripts/**/*.py` that calls `load_dataset`.
   - Add a `.env.example` + `scripts/data/_env.py` helper loaded at startup.
2. **M2**: Implement QK-norm in `src/modern_llm/models/attention.py`
   (two RMSNorms on Q,K after projection, before RoPE). Add `use_qk_norm`
   flag to `ModernLLMConfig`.
3. **M3**: Implement residual init scaling + scaled embeddings +
   z-loss in `src/modern_llm/models/transformer.py`.
4. **M4**: Implement streaming + packed dataloader in
   `src/modern_llm/data/lm_datasets.py`. Drop the
   `"Streaming datasets are not yet supported"` NotImplementedError.
5. **M5**: Add `torch.compile` toggle in `src/modern_llm/training/trainer_base.py`
   (behind a config flag, default ON for H100, OFF for RTX 3060).
6. **M6**: New config file `configs/lm_75m_h100.json` with the final
   Option-D architecture (see §1.2). Document in `configs/README.md`.
7. **M7**: Turn attention sinks OFF for training
   (`use_attention_sinks: false`) — keeps Flash-Attn fast path. Can be
   re-enabled for streaming inference.

### Phase B — Data (1 day eng + background download)

8. **M8**: `scripts/data/ensure_dataset.py` — symlink-or-download helper
   that checks `~/.cache/huggingface` first, targets `data/raw/`.
9. **M9**: `scripts/data/download_pretrain_mix.py` — downloads all 7
   pretrain datasets per §3 Stage 1 table. Runs in background.
10. **M10**: `scripts/data/tokenize_pretrain.py` — streams through the
    mix, tokenizes with cl100k, writes packed uint32 shards to
    `data/tokenized/pretrain_mix/shard_*.bin`. Uses weighted sampling per
    §3.
11. **M11**: `scripts/data/download_sft_mix.py` +
    `tokenize_sft.py` — ChatML formatting, loss mask encoded in a
    parallel `mask_*.bin`.
12. **M12**: Same for DPO (`download_dpo_mix.py`) and reasoning
    (`download_reasoning_mix.py`).

### Phase C — Stage runs (≈7 days wall clock)

13. **M13**: **Pretrain run** (Stage 1). Launch via
    `python scripts/run_pipeline.py --config configs/lm_75m_h100.json --stage pretrain`.
    Monitor loss + evals at 2k step cadence.
14. **M14**: **Anneal run** (Stage 2). Loads `pretrain_final.pt`,
    switches to anneal config + anneal mix.
15. **M15**: **SFT run** (Stage 3). ChatML formatting, Tulu-3 + SmolTalk
    + math mix.
16. **M16**: **Full eval** (16 benchmarks) on `sft_final.pt`. Gate.
17. **M17**: **DPO run** (Stage 4). UltraFeedback + HelpSteer2 +
    Skywork. Apply length-normalized DPO.
18. **M18**: **Reasoning SFT** (Stage 5A). OpenThoughts + NuminaMath +
    GSM8K-aug.
19. **M19**: (Conditional) **Verifier + rejection sampling** (Stage 5B).
20. **M20**: **Final full eval**. Update `stage_gains.md` +
    `baseline_comparison.md` with all 16 benchmarks.

### Phase D — Analysis & ship

21. **M21**: Write `experiments/results/final_report.md` comparing each
    stage checkpoint against GPT-2 small, GPT-2 medium, and SmolLM-2-135M
    across all 16 benchmarks. Include per-subject breakdowns for MMLU
    and MMLU-Pro.
22. **M22**: Tag the final checkpoint as
    `experiments/runs/lm-75m-h100/reasoning_final.pt` and push a
    README.md next to it documenting architecture, training, and evals.

---

## 10. Open questions (resolve before M13)

1. **Do we have the H100?** Confirm; if only A100 or RTX 3060, timelines
   in §6.3 roughly double on A100 and become infeasible on 3060 alone.
2. **License compatibility** for the final model checkpoint — if we
   include The Stack v2 we inherit its license conditions; confirm this
   is acceptable for the use case.
3. **Whether to retokenize (Option A).** Default answer: no. Revisit
   only after M16 if we are <27% MMLU.
4. **Do we want to include a safety/red-team dataset** in SFT? Tulu-3
   already has some; if a stronger safety signal is required add
   `HuggingFaceH4/no_robots` and `Anthropic/hh-rlhf` (already used).

---

**End of plan.**
