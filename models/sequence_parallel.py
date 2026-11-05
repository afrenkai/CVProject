import torch.distributed as dist
import torch.distributed._functional_collectives as functional
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup


def shard_sequence(tokens: Tensor, group: ProcessGroup) -> Tensor:
    world = dist.get_world_size(group)
    if tokens.shape[1] % world:
        raise ValueError(f"Sequence length {tokens.shape[1]} is not divisible by {world} sequence-parallel ranks")
    return tokens.chunk(world, dim=1)[dist.get_rank(group)]


def gather_sequence(tokens: Tensor, group: ProcessGroup) -> Tensor:
    return functional.all_gather_tensor(tokens.contiguous(), gather_dim=1, group=group)


def sequence_to_heads(tokens: Tensor, group: ProcessGroup) -> Tensor:
    world = dist.get_world_size(group)
    if tokens.shape[1] % world:
        raise ValueError(f"{tokens.shape[1]} heads are not divisible by {world} sequence-parallel ranks")
    outgoing = rearrange(tokens, "batch (rank head) token channel -> rank batch head token channel", rank=world)
    incoming = functional.all_to_all_single(outgoing.contiguous(), None, None, group)
    return rearrange(incoming, "rank batch head token channel -> batch head (rank token) channel")


def heads_to_sequence(tokens: Tensor, group: ProcessGroup) -> Tensor:
    world = dist.get_world_size(group)
    outgoing = rearrange(tokens, "batch head (rank token) channel -> rank batch head token channel", rank=world)
    incoming = functional.all_to_all_single(outgoing.contiguous(), None, None, group)
    return rearrange(incoming, "rank batch head token channel -> batch (rank head) token channel")
