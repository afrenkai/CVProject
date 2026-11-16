from transformers import PretrainedConfig


class SpeakUpConfig(PretrainedConfig):
    model_type = "speakup"

    def __init__(
        self,
        encoder: dict[str, object] | None = None,
        model: dict[str, object] | None = None,
        objective: dict[str, object] | None = None,
        encoder_bfloat16: bool = False,
        **kwargs: object,
    ) -> None:
        self.encoder = encoder or {}
        self.model = model or {}
        self.objective = objective or {}
        self.encoder_bfloat16 = encoder_bfloat16
        super().__init__(**kwargs)
