import torch


def cuda_device(seed: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("AV training requires CUDA FP32")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda", torch.cuda.current_device())
