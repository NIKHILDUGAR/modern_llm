"""Training utilities and entrypoints.

Submodules are imported explicitly by callers (e.g.
`from modern_llm.training.train_lm import run_training`) to avoid a
circular import: `lm_datasets` imports `modern_llm.training.distributed`,
and eagerly pulling `train_lm` here would force `modern_llm.data` to
initialize before its own __init__ finishes when the entry point imports
`modern_llm.data.*` first.
"""

