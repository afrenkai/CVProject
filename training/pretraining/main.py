import os
import time
from datetime import UTC, datetime
from pathlib import Path
import hydra
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, set_seed
from architecture.pretraining.joint_encoder import JointEncoder
from train_datasets.online_corpus import OnlineCorpusWindows
from train_datasets.records import read_windows
from training.pretraining.allocation import (
    AllocationProgress,
    allocation_deadline,
)
from training.pretraining.configuration import PretrainingConfig
from training.pretraining.frozen import build_online_objective
from training.pretraining.reporting import PretrainingMetrics
from training.pretraining.schedule import (
    ValidationRequests,
    native_checkpoint_step,
    pretraining_arguments,
)


def resume_step(config: PretrainingConfig, world_size: int) -> int:
    checkpoint = config.runtime.resume
    if checkpoint is None:
        assert not list(config.output.glob("checkpoint-*")), "Choose an explicit native checkpoint to resume"
        return 0
    assert checkpoint.resolve().parent == config.output.resolve()
    return native_checkpoint_step(checkpoint, world_size)


def finish_allocation(
    trainer: Trainer, config: PretrainingConfig, phase: Path, start: int, training_seconds: float
) -> None:
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    end = trainer.state.global_step
    assert start < end <= (config.runtime.pause_step or config.trainer.maximum_steps)
    checkpoint = config.output / f"checkpoint-{end}"
    assert native_checkpoint_step(checkpoint, trainer.args.world_size) == end
    if trainer.is_world_process_zero():
        report = {
            "completed": True,
            "global_start": start,
            "global_end": end,
            "global_step_cap": config.trainer.maximum_steps,
            "optimizer_updates": end - start,
            "training_seconds": training_seconds,
            "seconds_per_optimizer_update": training_seconds / (end - start),
            "checkpoint": str(checkpoint),
            "checkpoint_retention": "all",
            "validation_due": end % config.schedule.validation_steps == 0,
            "validation_blocks_training": config.schedule.validation_blocks_training,
            "normalizer": str(config.normalization.latents),
            "test_cs_r6_used": False,
        }
        (phase / "completed.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        print(yaml.safe_dump(report, sort_keys=False), flush=True)
    trainer.accelerator.wait_for_everyone()


def run_pretraining(config: PretrainingConfig, source_config: DictConfig) -> None:
    set_seed(config.trainer.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arguments = pretraining_arguments(config)
    device = arguments.device
    assert device.type == "cuda" and torch.cuda.is_bf16_supported()
    assert device.index == int(os.environ.get("LOCAL_RANK", "0"))
    assert arguments.world_size == torch.cuda.device_count()
    assert config.data.context_frames == config.encoder.context_frames
    deadline = allocation_deadline(config.runtime)
    assert deadline is None or time.time() < deadline
    assert config.runtime.stop_file is None or not config.runtime.stop_file.exists()
    start = resume_step(config, arguments.world_size)
    pause = config.runtime.pause_step or config.trainer.maximum_steps
    assert start < pause <= config.trainer.maximum_steps
    training = read_windows(config.windows.train, "train")
    validation = read_windows(config.windows.validation, "validation")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    allocation_id = os.environ.get("SLURM_JOB_ID", os.environ.get("TORCHELASTIC_RUN_ID", f"local_{timestamp}"))
    phase = config.output / "allocations" / allocation_id
    phase.mkdir(parents=True, exist_ok=True)
    assert not (phase / "completed.yaml").exists()
    resolved = phase / "config.yaml"
    if arguments.process_index == 0:
        OmegaConf.save(source_config, resolved, resolve=True)
    encoder = JointEncoder(config.encoder, device)
    objective = build_online_objective(config, encoder.encode, device)
    trainer = Trainer(
        model=objective,
        args=arguments,
        train_dataset=OnlineCorpusWindows(
            training, config.data, config.output / "input_events", config.trainer.seed, config.trainer.raw_read_attempts
        ),
        callbacks=[
            PretrainingMetrics(phase),
            ValidationRequests(
                config.schedule, resolved.resolve(), config.output / "validation_requests", len(validation)
            ),
        ],
    )
    trainer.add_callback(AllocationProgress(trainer.accelerator, start, pause, deadline, config.runtime.stop_file))
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=None if config.runtime.resume is None else str(config.runtime.resume))
    finish_allocation(trainer, config, phase, start, time.monotonic() - started)
    trainer.accelerator.end_training()


@hydra.main(config_path="../../configs", config_name="pretraining", version_base=None)
def main(config: DictConfig) -> None:
    settings = instantiate(config)
    assert isinstance(settings, PretrainingConfig)
    run_pretraining(settings, config)


if __name__ == "__main__":
    main()
