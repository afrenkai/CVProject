from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


class ControlLayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keyboard_count: PositiveInt = 256
    mouse_buttons: tuple[str, ...] = ("left", "right", "middle", "x1", "x2")
    axis_names: tuple[str, ...] = ("raw_mouse_x", "raw_mouse_y", "wheel", "horizontal_wheel")

    @property
    def dimension(self) -> int:
        return self.keyboard_count + len(self.mouse_buttons) + len(self.axis_names)

# got this from loading parquet and seeing what everything maps to
class Opencs2InputMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keys: dict[str, tuple[int, ...]] = {
        "attack": (1,),
        "attack2": (2,),
        "back": (83,),
        "duck": (17, 162),
        "forward": (87,),
        "jump": (32,),
        "look_at_weapon": (70,),
        "move_left": (65,),
        "move_right": (68,),
        "reload": (82,),
        "score": (9,),
        "speed": (16, 160),
        "use": (69,),
    }
    buttons: dict[str, str] = {"attack": "left", "attack2": "right"}
    degrees_per_count: PositiveFloat = 0.022
    minimum_ticks_per_block: PositiveInt = 4


class ControlEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout: ControlLayout = Field(default_factory=ControlLayout)
    axis_scales: tuple[PositiveFloat, ...]
    hidden_dimension: PositiveInt = 384


class MediaReference(BaseModel):
    uri: str
    pts_ns: int


class ScreenPayload(BaseModel):
    utc_ns: int
    media_ref: MediaReference


class ActionPayload(BaseModel):
    buttons: tuple[int | str, ...] = ()
    event_type: str | None = None
    vk: int | None = Field(default=None, ge=0, lt=256)
    button: str | None = None
    pressed: bool | None = None
    dx: int = 0
    dy: int = 0
    last_x: int = 0
    last_y: int = 0
    us_flags: int = 0
    timestamp: int | None = None


@dataclass(frozen=True)
class RecordedEvent:
    time_ns: int
    topic: str
    payload: ActionPayload


@dataclass(frozen=True)
class ControlCapture:
    events: list[RecordedEvent]
    screen_times_ns: list[int]
    initial_state_ns: int
    clock_origin_ns: int
    maximum_residual_ns: int
    maximum_delivery_delay_ns: int
    duplicate_screen_references: int


@dataclass
class CaptureMessages:
    events: list[RecordedEvent] = field(default_factory=list)
    screen_times_ns: list[int] = field(default_factory=list)
    origins: set[int] = field(default_factory=set)
    raw_origins: list[int] = field(default_factory=list)
    delivery_delays: list[int] = field(default_factory=list)
    state_times: dict[str, int] = field(default_factory=dict)
    duplicate_screen_references: int = 0
