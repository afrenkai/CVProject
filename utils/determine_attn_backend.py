import torch
from models.model_types import AttentionBackend


def figure_out_backend() -> AttentionBackend:
    """
    insanely stupid 3am script to determine which attn backend to use based on hardware
    mayb someth built in, but this took 30s to write
    """
    if not torch.cuda.is_available():
        return 'sdpa'
    else:

        device = torch.cuda.current_device()
        match torch.cuda.get_device_capability(device)[0]:
            case 8:
                return 'flex'
            case 9:
                #on hopper, can use fa4 + flexattn
                return 'flex_flash'
            case 10:
                return 'flex'
            case 11:
                return 'sdpa'
            case 12:
                return 'flex'
            case _:
                return 'sdpa'

if __name__ == "__main__":
    print(figure_out_backend())

