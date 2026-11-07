import bisect
import math
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from data.control_types import ControlCapture, RecordedEvent
from data.data_types import ActionWindowConfig


def update_held_state(event: RecordedEvent, keys: set[int], buttons: set[str], settings: ActionWindowConfig) -> None:
    payload = event.payload
    layout = settings.control_layout
    match event.topic, payload.event_type:
        case "keyboard/state", str() | None:
            assert all(isinstance(key, int) and 0 <= key < layout.keyboard_count for key in payload.buttons)
            keys.clear()
            keys.update(int(key) for key in payload.buttons)
        case "mouse/state", str() | None:
            assert all(str(button) in layout.mouse_buttons for button in payload.buttons)
            buttons.clear()
            buttons.update(str(button) for button in payload.buttons)
        case "keyboard", "press" if payload.vk is not None:
            keys.add(payload.vk)
        case "keyboard", "release" if payload.vk is not None:
            keys.discard(payload.vk)
        case "keyboard", event_type:
            raise AssertionError(f"Malformed keyboard event {event_type} vk={payload.vk}")
        case "mouse", "click":
            assert payload.button in layout.mouse_buttons and payload.pressed is not None
            if payload.pressed:
                buttons.add(str(payload.button))
            else:
                buttons.discard(str(payload.button))


def add_held_duration(
    row: Tensor, elapsed_ns: int, keys: set[int], buttons: set[str], settings: ActionWindowConfig
) -> None:
    assert elapsed_ns >= 0
    for key in keys:
        row[key] += elapsed_ns
    for button in buttons:
        row[settings.control_layout.keyboard_count + settings.control_layout.mouse_buttons.index(button)] += elapsed_ns


@jaxtyped(typechecker=beartype)
def controls_for_window(
    capture: ControlCapture, start_seconds: float, settings: ActionWindowConfig
) -> Float[Tensor, "block control"]:
    assert math.isfinite(start_seconds)
    start_ns = round(start_seconds * 1e9)
    block_ns = round(1e9 * settings.video_frames_per_step / settings.video_rate)
    block_count = (settings.context_frames + settings.target_frames) // settings.video_frames_per_step
    end_ns = start_ns + block_count * block_ns
    assert capture.initial_state_ns < start_ns
    assert capture.screen_times_ns[0] <= start_ns < end_ns <= capture.screen_times_ns[-1]
    left = max(0, bisect.bisect_right(capture.screen_times_ns, start_ns) - 1)
    right = min(len(capture.screen_times_ns) - 1, bisect.bisect_left(capture.screen_times_ns, end_ns))
    assert all(
        capture.screen_times_ns[index + 1] - capture.screen_times_ns[index] <= settings.maximum_screen_gap * 1e9
        for index in range(left, right)
    ), "Screen capture gaps exceed the source-clock contract"
    keys: set[int] = set()
    buttons: set[str] = set()
    event_index = 0
    while event_index < len(capture.events) and capture.events[event_index].time_ns < start_ns:
        update_held_state(capture.events[event_index], keys, buttons, settings)
        event_index += 1
    held_count = settings.control_layout.keyboard_count + len(settings.control_layout.mouse_buttons)
    controls = torch.zeros(block_count, settings.control_layout.dimension, dtype=torch.float64)
    raw_x = 0
    raw_y = 0
    for block in range(block_count):
        cursor = start_ns + block * block_ns
        stop = cursor + block_ns
        while event_index < len(capture.events) and capture.events[event_index].time_ns < stop:
            event = capture.events[event_index]
            assert cursor <= event.time_ns < stop
            add_held_duration(controls[block], event.time_ns - cursor, keys, buttons, settings)
            cursor = event.time_ns
            update_held_state(event, keys, buttons, settings)
            if event.topic == "mouse/raw":
                controls[block, held_count] += event.payload.last_x
                controls[block, held_count + 1] += event.payload.last_y
                raw_x += event.payload.last_x
                raw_y += event.payload.last_y
            elif event.topic == "mouse" and event.payload.event_type == "scroll":
                controls[block, held_count + 2] += event.payload.dy
                controls[block, held_count + 3] += event.payload.dx
            event_index += 1
        add_held_duration(controls[block], stop - cursor, keys, buttons, settings)
        controls[block, :held_count] /= block_ns
    assert controls[:, :held_count].min() >= 0 and controls[:, :held_count].max() <= 1
    assert controls[:, held_count].sum().item() == raw_x and controls[:, held_count + 1].sum().item() == raw_y
    assert torch.isfinite(controls).all()
    return controls.float()
