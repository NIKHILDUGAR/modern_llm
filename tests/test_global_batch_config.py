from modern_llm.config.pipeline_config import PipelineConfig
from modern_llm.training.distributed import scale_grad_accum_for_world_size


def test_scale_grad_accum_for_world_size_respects_world_size(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert scale_grad_accum_for_world_size(32, 4) == 4


def test_pipeline_config_uses_global_batch_semantics(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = PipelineConfig(pretrain_batch_size=32, pretrain_micro_batch_size=4)
    train_config = config.get_pretrain_config()
    assert train_config.gradient_accumulation_steps == 4


def test_pipeline_config_preserves_single_gpu_behavior(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = PipelineConfig(pretrain_batch_size=32, pretrain_micro_batch_size=4)
    train_config = config.get_pretrain_config()
    assert train_config.gradient_accumulation_steps == 8
