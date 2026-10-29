from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FiniteModel(FrozenModel):
    model_config = ConfigDict(allow_inf_nan=False)
