from torch import Tensor, nn

type StateDictionary = dict[str, Tensor]
type Checkpoint = dict[str, StateDictionary]
type ModelModules = dict[str, nn.Module]
