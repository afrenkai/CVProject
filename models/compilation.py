import torch
from torch import nn
from models.transformers import AdaptiveAttentionBlock


def set_recompile_limit(limit: int) -> None:
    torch._dynamo.config.recompile_limit = limit
    torch._dynamo.config.accumulated_recompile_limit = max(limit, torch._dynamo.config.accumulated_recompile_limit)


def compile_blocks(module: nn.Module) -> int:
    blocks = [child for child in module.modules() if isinstance(child, AdaptiveAttentionBlock)]
    for block in blocks:
        block.compile(dynamic=False)
    return len(blocks)
