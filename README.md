# Modern LLM
The README is to be updated post achieving its goal in training.
The following is the old README from the ORIGINAL PROJECT, This current version has no updated README as of yet hence keeping this here for some understanding of what the project contains. 

A from-scratch implementation of a **frontier-style LLM training pipeline**, demonstrating modern architectural choices (RoPE, RMSNorm, SwiGLU, attention sinks) and a complete alignment workflow (Pretrain → SFT → DPO → Verifier). Trained a 253M parameter model that achieves **27.03 perplexity** on WikiText-2, outperforming GPT-2 (124M) at 40.64.

Read the detailed documentation here: https://aymanmahfuz27-modern_llm.mintlify.app/architecture/overview


---

## Results

### Perplexity (WikiText-2)

| Model | Parameters | PPL | vs GPT-2 |
|-------|------------|-----|----------|
| GPT-2 (baseline) | 124M | 40.64 | — |
| **Ours (pretrain)** | 253M | **27.03** | **-33%** |
| Ours (SFT) | 253M | 34.14 | -16% |
| Ours (DPO) | 253M | 34.32 | -16% |

### Task Metrics (Few-Shot)

| Model | SST-2 Acc | GSM8K EM |
|-------|-----------|----------|
| GPT-2 | 56.0% | N/A |
| Ours (pretrain) | 49.5% | 0.0% |
| Ours (SFT) | 53.5% | 0.0% |
| Ours (DPO) | 49.5% | 0.0% |

*Note: Task metrics below baseline due to training data distribution differences and prompt sensitivity. GSM8K 0% is expected for 253M models. Full analysis in `report/final_tacc_report.md`.*

---

## Architecture Deep-Dive

The model implements a PaLM/LLaMA-style decoder-only Transformer with four key modern components:

### RMSNorm

**Reference:** Zhang & Sennrich, 2019

```
y = x · γ / √(mean(x²) + ε)
```

Faster than LayerNorm (no mean subtraction), used in LLaMA/PaLM. Stabilizes training without centering.

```19:55:src/modern_llm/models/layers.py
class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019).

    Math:
        y = x * γ / sqrt(mean(x^2) + ε)
        where γ is a learned weight vector.
    ...
    def forward(self, x: Tensor) -> Tensor:
        # mean(x^2) is the RMS statistic from Zhang & Sennrich (2019, Eq. 3)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight
```

### RoPE (Rotary Position Embeddings)

**Reference:** Su et al., 2021

```
q' = q ⊙ cos(mθ) + rotate_half(q) ⊙ sin(mθ)
```

Encodes relative positions via rotation matrices applied to Q/K. Better length extrapolation than absolute embeddings.

```190:196:src/modern_llm/models/attention.py
    def _apply_rope(self, tensor: Tensor, seq_len: int, offset: int = 0) -> Tensor:
        cos, sin = self._get_rope_factors(seq_len + offset, tensor.device, tensor.dtype)
        cos = cos[offset : offset + seq_len]
        sin = sin[offset : offset + seq_len]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (tensor * cos) + (self._rotate_half(tensor) * sin)
```

### SwiGLU

**Reference:** Shazeer, 2020; Chowdhery et al., 2022

```
SwiGLU(x) = (Wg·x ⊙ swish(Wv·x)) · Wo
```

Gated linear unit with Swish activation. 2-4% better than GELU with similar parameter count.

```72:114:src/modern_llm/models/layers.py
class SwiGLU(nn.Module):
    """SwiGLU feedforward block (Shazeer, 2020; Chowdhery et al., 2022).

    Math:
        SwiGLU(x) = W_o[(W_g x) ⊙ swish(W_v x)]
        where W_g splits into gate/value projections and ⊙ is element-wise multiply.
    ...
    def forward(self, x: Tensor) -> Tensor:
        gate_out, value = self.gate(x).chunk(2, dim=-1)
        activated = _swish(gate_out)
        gated = activated * value
        return self.proj(gated)
```

### Attention Sinks

**Reference:** Press et al., 2021; Xiao et al., 2023

Learnable "sink" tokens that every position can attend to, stabilizing generation beyond the training context length.

```95:100:src/modern_llm/models/attention.py
        if config.use_attention_sinks:
            self.sink_states = nn.Parameter(
                torch.randn(config.num_attention_sinks, config.d_model) * 0.02
            )
        else:
            self.register_parameter("sink_states", None)
```

---

## Training Pipeline

```
WikiText-103 + TinyStories (600M tokens)
         ↓
    [Pretrain] → Base Model (253M params, d=768, L=12, H=12)
         ↓
    [SFT] → Instruction-tuned (Alpaca 52K)
         ↓
    [DPO] → Preference-aligned (HH-RLHF 161K)
         ↓
    [Verifier] → Answer scoring (GSM8K)
```

**References:**
- SFT: Ouyang et al., 2022 (InstructGPT)
- DPO: Rafailov et al., 2023
- Verifier: Lightman et al., 2023

---

## Code Organization

```
modern_llm/
├── checkpoints/                    # Trained model checkpoints
│   ├── pretrain_best.pt            # Base language model
│   ├── sft_final.pt                # Instruction-tuned
│   ├── dpo_final.pt                # Preference-aligned
│   └── verifier_final.pt           # Answer scoring model
│
├── configs/                        # Training configs
│   └── lm_max_rtx3060.json         # Max model for RTX 3060
│
├── experiments/
│   ├── results/                    # Evaluation JSONs/CSVs
│   ├── runs/                       # Training checkpoints
│   └── comparison_log_v2.txt       # Key perplexity results
│
├── report/                         # Generated markdown reports
│
├── scripts/
│   ├── run_pipeline.py             # Unified Python entry point
│   ├── run_pipeline.py             # Full pipeline orchestrator (stages or full)
│   ├── evaluate_and_compare.py     # GPT-2 comparison script
│   ├── pretrain.py                 # Standalone pretraining
│   ├── sft.py                      # Standalone SFT
│   ├── dpo.py                      # Standalone DPO
│   └── train_verifier.py           # Verifier training
│
├── src/modern_llm/
│   ├── models/
│   │   ├── transformer.py          # ModernDecoderLM
│   │   ├── attention.py            # Multi-head attention + RoPE + sinks
│   │   ├── layers.py               # RMSNorm, SwiGLU
│   │   ├── moe.py                  # Mixture-of-Experts (optional)
│   │   └── verifier.py             # Answer correctness model
│   │
│   ├── config/
│   │   ├── model_config.py         # ModernLLMConfig, MoEConfig
│   │   ├── train_config.py         # TrainingConfig
│   │   ├── hardware_config.py      # Hardware presets
│   │   └── pipeline_config.py      # Full pipeline config
│   │
│   ├── data/
│   │   ├── lm_datasets.py          # Language modeling data
│   │   ├── instruction_datasets.py # SFT instruction data
│   │   └── preference_datasets.py  # DPO preference pairs
│   │
│   ├── training/
│   │   ├── trainer_base.py         # Shared Trainer (AMP, grad accum)
│   │   ├── train_lm.py             # Pretrain entrypoint
│   │   ├── train_sft.py            # SFT entrypoint
│   │   ├── train_dpo.py            # DPO entrypoint
│   │   └── train_verifier.py       # Verifier entrypoint
│   │
│   ├── alignment/
│   │   ├── dpo_loss.py             # DPO objective
│   │   └── alignment_pipeline.py   # Pipeline orchestrator
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Perplexity, accuracy, ROUGE
│   │   └── pipeline_eval.py        # Stage comparison
│   │
│   └── utils/
│       ├── checkpointing.py        # Save/load checkpoints
│       └── logging_utils.py        # Logging setup
│
├── tests/                          # Unit tests
├── scripts/run_pipeline.py         # One-button entry point
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/modern_llm.git
cd modern_llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify installation
python -c "from modern_llm.models import ModernDecoderLM; print('OK')"

# Smoke test (5 minutes, CPU/GPU)
python scripts/run_pipeline.py --config local-smoke --stage all
```

---

## Using Pre-trained Checkpoints

```bash
# Verify checkpoints load correctly
python scripts/verify_checkpoints.py

# Run evaluation on existing checkpoints
python scripts/evaluate_and_compare.py

# Run GSM8K math benchmark with verifier
python scripts/benchmark_gsm8k.py
```

---

## Running Training (from scratch)

```bash
# Full pipeline (~24 hours on RTX 3060)
python scripts/run_pipeline.py --config local --stage all

# Individual stages
python scripts/run_pipeline.py --config local --stage pretrain
python scripts/run_pipeline.py --config local --stage sft --checkpoint checkpoints/pretrain_best.pt
python scripts/run_pipeline.py --config local --stage dpo --checkpoint checkpoints/sft_final.pt
```

### Config Presets

| Preset | Hardware | Duration | Description |
|--------|----------|----------|-------------|
| `local-smoke` | Any | ~5 min | Quick sanity check |
| `local` | RTX 3060 | ~24 hours | Full training |

---

## Reproducing Results

The results in `experiments/comparison_log_v2.txt` can be reproduced with:

```bash
# Evaluate pre-trained checkpoints against GPT-2
python scripts/evaluate_and_compare.py

# Or run full training from scratch
python scripts/run_pipeline.py --config local --stage all
```

---

## References

- **RMSNorm:** Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *NeurIPS*.
- **RoPE:** Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*.
- **SwiGLU:** Shazeer, N. (2020). GLU Variants Improve Transformer. *arXiv:2002.05202*.
- **Attention Sinks:** Xiao, G., et al. (2023). Efficient Streaming Language Models with Attention Sinks. *arXiv:2309.17453*.
- **InstructGPT:** Ouyang, L., et al. (2022). Training language models to follow instructions. *NeurIPS*.
- **DPO:** Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS*.
- **Verifier:** Lightman, H., et al. (2023). Let's Verify Step by Step. *arXiv:2305.20050*.

---

## License

MIT
