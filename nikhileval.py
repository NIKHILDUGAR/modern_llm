import torch
from pathlib import Path
from scripts.evaluation.eval_sst2 import (
    load_scratch_model,
    evaluate_sst2,
)
from scripts.evaluation.eval_gsm8k import (
    evaluate_gsm8k,
    load_verifier,
)
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
bench='gpqa,hle,ifbench_test,ifeval,mixeval_easy,mmlu_pro,anli,bbq,commonsenseqa,coqa,glue,hellaswag,mmlu,squad_v2'
import torch.nn as nn
def print_model_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Percentage Trainable: {100 * trainable_params / total_params:.4f}%\n")
    
    # Optional: Print trainable layers specifically
    print("Trainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}: {param.shape}")



evaluation_tracker = EvaluationTracker(output_dir="./resultsnikhil")
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    max_samples=2
)

def evaluate_model_comprehensive(
    checkpoint_path: str,
    verifier_path: str = None,
    output_dir: str = "experiments/results",
):
    """Run all evaluations for a model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model: {checkpoint_path}")
    model, tokenizer = load_scratch_model(checkpoint_path, device)
    pipeline = Pipeline(
    model=model,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    tasks=bench,
)   
    model.eval()
    print_model_parameters(model)
    results = pipeline.evaluate()
    pipeline.show_results()
    results = pipeline.get_results() 
    print(f"\nResults saved to {output_dir}/evaluation_results.json")
    return results

# Run evaluation
evaluate_model_comprehensive(
    checkpoint_path="experiments/runs/gpu-full/gpu-full-sft//gpu-full-sft_final.pt",
)
