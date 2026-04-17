"""Instruction-tuning datasets for SFT (e.g., Alpaca, OpenAssistant).

Implements standardized loading and formatting for instruction-following
datasets, with a unified chat template for consistency across SFT and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from modern_llm.training.distributed import maybe_distributed_sampler
from modern_llm.utils.paths import apply_env_defaults, cache_dir_for_datasets

apply_env_defaults()


@dataclass
class InstructionDatasetConfig:
    """Configuration for instruction-tuning datasets."""

    dataset_name: str
    max_length: int = 1024
    split: str = "train"
    num_examples: Optional[int] = None  # Limit examples for debugging
    include_input: bool = True  # Whether to include the "input" field

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


# Chat template for instruction formatting
INSTRUCTION_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

INSTRUCTION_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


def format_instruction(instruction: str, input_text: str, output: str) -> str:
    """Format an instruction example using standard template.

    Pre: instruction and output are non-empty strings.
    Post: Returns formatted string with clear section delimiters.
    """
    if input_text and input_text.strip():
        return INSTRUCTION_TEMPLATE.format(
            instruction=instruction.strip(),
            input=input_text.strip(),
            output=output.strip(),
        )
    return INSTRUCTION_TEMPLATE_NO_INPUT.format(
        instruction=instruction.strip(),
        output=output.strip(),
    )


class InstructionDataset(Dataset):
    """PyTorch Dataset for instruction-tuning.

    Loads and tokenizes instruction-following data with masking so that
    only the response portion contributes to the loss.
    """

    def __init__(
        self,
        config: InstructionDatasetConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.config = config
        self.tokenizer = tokenizer

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.examples = self._load_and_process()

    def _load_and_process(self) -> list[dict]:
        """Load dataset from HF Hub and format as instruction examples."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("datasets package required: pip install datasets") from e

        raw = load_dataset(self.config.dataset_name, split=self.config.split, cache_dir=cache_dir_for_datasets())

        if self.config.num_examples:
            raw = raw.select(range(min(self.config.num_examples, len(raw))))

        examples = []
        for item in raw:
            formatted = self._format_example(item)
            if formatted:
                tokenized = self._tokenize(formatted)
                if tokenized:
                    examples.append(tokenized)

        return examples

    def _format_example(self, item: dict) -> Optional[str]:
        """Convert a raw dataset item to formatted instruction string.

        Handles both Alpaca format and OpenAssistant format.
        """
        # Alpaca format: instruction, input, output
        if "instruction" in item and "output" in item:
            return format_instruction(
                instruction=item["instruction"],
                input_text=item.get("input", "") if self.config.include_input else "",
                output=item["output"],
            )

        # OpenAssistant format: text (conversation turns)
        if "text" in item:
            return item["text"]

        # HH-RLHF format: chosen response (for SFT, we train on chosen)
        if "chosen" in item:
            return item["chosen"]

        return None

    def _tokenize(self, text: str) -> Optional[dict]:
        """Tokenize and create labels with response-only masking.

        Labels are -100 for prompt tokens so only response tokens
        contribute to the cross-entropy loss.
        """
        # Find where the response starts
        response_marker = "### Response:\n"
        response_start = text.find(response_marker)

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Create labels - mask everything before response
        labels = input_ids.clone()

        if response_start != -1:
            # Find token index corresponding to response start
            prompt_text = text[:response_start + len(response_marker)]
            prompt_tokens = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            prompt_len = prompt_tokens["attention_mask"].sum().item()

            # Mask prompt tokens in labels
            labels[:prompt_len] = -100

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def load_instruction_dataset(
    config: InstructionDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> InstructionDataset:
    """Factory function for loading instruction datasets.

    Pre: tokenizer has pad_token defined (or will use eos_token).
    Post: Returns InstructionDataset with tokenized examples.
    """
    return InstructionDataset(config, tokenizer)


def create_instruction_dataloader(
    dataset: InstructionDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for instruction dataset.

    Pre: dataset is a valid InstructionDataset.
    Post: Returns DataLoader with proper collation.
    """

    def collate_fn(batch: list[dict]) -> dict:
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }

    sampler = maybe_distributed_sampler(dataset, shuffle=shuffle)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )



