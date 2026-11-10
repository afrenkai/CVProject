from collections.abc import Callable
import torch
from gram_newton_schulz import Muon as GramMuon
from gram_newton_schulz_ampere import Dion3
from gram_newton_schulz_ampere import Muon as AmpereGramMuon
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_with_min_lr_schedule_with_warmup
from models.transformers import AdaptiveAttentionBlock
from training.training_types import OptimizerConfig


class OptimizerCollection(Optimizer):
    def __init__(self, optimizers: list[Optimizer]) -> None:
        parameters = [
            parameter for optimizer in optimizers for group in optimizer.param_groups for parameter in group["params"]
        ]
        super().__init__(parameters, {})
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    def step(self, closure: Callable[[], float] | None = None) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, object]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        states = state_dict["optimizers"]
        assert len(states) == len(self.optimizers)
        for index, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(states[index])
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]


def build_optimizer(objective: nn.Module, config: OptimizerConfig) -> OptimizerCollection:
    learning_rate, weight_decay = config.learning_rate, config.weight_decay
    trainable = [(name, parameter) for name, parameter in objective.named_parameters() if parameter.requires_grad]
    block_matrices = [
        parameter
        for module in objective.modules()
        if isinstance(module, AdaptiveAttentionBlock)
        for parameter in module.parameters()
        if parameter.ndim == 2 and parameter.requires_grad
    ]
    block_matrix_ids = {id(parameter) for parameter in block_matrices}
    remaining = [parameter for name, parameter in trainable if id(parameter) not in block_matrix_ids]
    match config.name:
        case "adamw":
            decayed = [parameter for name, parameter in trainable if parameter.ndim > 1 and "bias" not in name]
            exempt = [parameter for name, parameter in trainable if parameter.ndim == 1 or "bias" in name]
            optimizers = [
                torch.optim.AdamW(
                    [{"params": decayed, "weight_decay": weight_decay}, {"params": exempt, "weight_decay": 0.0}],
                    lr=learning_rate,
                    betas=config.betas,
                )
            ]
        case "muon":
            optimizers = [
                torch.optim.Muon(
                    block_matrices, lr=learning_rate, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"
                ),
                torch.optim.AdamW(remaining, lr=learning_rate, weight_decay=weight_decay),
            ]
        case "gns_muon":
            optimizers = [
                GramMuon(
                    block_matrices,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    adjust_lr="rms_norm",
                    ns_algorithm="gram_newton_schulz",
                ),
                torch.optim.AdamW(remaining, lr=learning_rate, weight_decay=weight_decay),
            ]
        case "gns_muon_ampere":
            optimizers = [
                AmpereGramMuon(
                    [{"params": block_matrices, "algorithm": "muon"}, {"params": remaining, "algorithm": "adamw"}],
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    adjust_lr="rms_norm",
                    ns_algorithm="gram_newton_schulz",
                )
            ]
        case "dion3":
            optimizers = [
                Dion3(
                    [{"params": block_matrices, "algorithm": "dion3"}, {"params": remaining, "algorithm": "adamw"}],
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    adjust_lr="rms_norm",
                )
            ]
    return OptimizerCollection(optimizers)


def warmup_ema_update(decay: float) -> Callable[[list[Tensor], list[Tensor], Tensor], None]:
    def update(averaged: list[Tensor], current: list[Tensor], count: Tensor) -> None:
        updates = float(count)
        rate = min(decay, (1.0 + updates) / (10.0 + updates))
        for index in range(len(averaged)):
            averaged[index].lerp_(current[index], 1.0 - rate)

    return update


def build_schedule(optimizer: Optimizer, config: OptimizerConfig, total_steps: int) -> LambdaLR:
    return get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps, min_lr_rate=config.final_learning_rate_fraction
    )
