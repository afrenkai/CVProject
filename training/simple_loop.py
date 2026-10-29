import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from training.pretraining.objective import PretrainingObjective


def train_epoch(
    objective: PretrainingObjective,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float]:
    objective.train()
    totals: dict[str, float] = {}
    n = 0
    for batch in dataloader:
        frames = batch["frames"].to(device)
        waveform = batch["waveform"].to(device)
        actions = batch["actions"].to(device)
        action_present = batch["action_present"].to(device)
        example_seed = batch["example_seed"].to(device)
        optimizer.zero_grad()
        out = objective(frames, waveform, actions, action_present, example_seed)
        out["loss"].backward()
        optimizer.step()
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    objective: PretrainingObjective,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    objective.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch in dataloader:
        frames = batch["frames"].to(device)
        waveform = batch["waveform"].to(device)
        actions = batch["actions"].to(device)
        action_present = batch["action_present"].to(device)
        example_seed = batch["example_seed"].to(device)
        out = objective(frames, waveform, actions, action_present, example_seed)
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}
