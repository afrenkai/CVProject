from pathlib import Path
from mcap.reader import make_reader
from data.control_types import (
    ActionPayload,
    CaptureMessages,
    ControlCapture,
    RecordedEvent,
    ScreenPayload,
)
from data.data_types import ActionClipConfig


def add_screen_reference(messages: CaptureMessages, source: Path, screen: ScreenPayload, log_time: int) -> None:
    assert Path(screen.media_ref.uri).name == source.name
    if messages.screen_times_ns:
        assert screen.media_ref.pts_ns >= messages.screen_times_ns[-1]
    if messages.screen_times_ns and screen.media_ref.pts_ns == messages.screen_times_ns[-1]:
        assert screen.utc_ns - screen.media_ref.pts_ns in messages.origins
        messages.duplicate_screen_references += 1
    else:
        messages.screen_times_ns.append(screen.media_ref.pts_ns)
    messages.origins.add(screen.utc_ns - screen.media_ref.pts_ns)
    delivery_delay = log_time - screen.media_ref.pts_ns
    assert delivery_delay >= 0
    messages.delivery_delays.append(delivery_delay)


def add_control_event(messages: CaptureMessages, topic: str, payload: ActionPayload, log_time: int) -> None:
    if topic == "mouse/raw":
        assert payload.us_flags == 0, "Only recorded relative raw mouse motion is supported"
        assert payload.timestamp is not None
        messages.raw_origins.append(payload.timestamp - log_time)
    if topic.endswith("/state"):
        messages.state_times.setdefault(topic, log_time)
    messages.events.append(RecordedEvent(log_time, topic, payload))


def finish_capture(messages: CaptureMessages, settings: ActionClipConfig) -> ControlCapture:
    assert len(messages.screen_times_ns) >= 3 and len(messages.origins) == 1
    assert set(messages.state_times) == {"keyboard/state", "mouse/state"}
    assert messages.raw_origins
    assert all(
        messages.screen_times_ns[index] < messages.screen_times_ns[index + 1]
        for index in range(len(messages.screen_times_ns) - 1)
    )
    origin = next(iter(messages.origins))
    maximum_residual = max(abs(origin - raw_origin) for raw_origin in messages.raw_origins)
    assert maximum_residual <= settings.maximum_clock_residual * 1e9, maximum_residual
    return ControlCapture(
        messages.events,
        messages.screen_times_ns,
        max(messages.state_times.values()),
        origin,
        maximum_residual,
        max(messages.delivery_delays),
        messages.duplicate_screen_references,
    )


def read_recorded_capture(
    source: Path, settings: ActionClipConfig, stop_seconds: float, start_seconds: float = 0.0
) -> ControlCapture:
    messages = CaptureMessages()
    topics = ["screen", "keyboard", "keyboard/state", "mouse", "mouse/state", "mouse/raw"]
    previous_time = -1
    with source.with_suffix(".mcap").open("rb") as stream:
        reader = make_reader(stream)
        assert reader.get_header().profile == "owa"
        for schema, channel, message in reader.iter_messages(
            topics=topics,
            log_time_order=True,
            start_time=round(start_seconds * 1e9),
            end_time=round(stop_seconds * 1e9),
        ):
            assert schema is not None and schema.encoding == "jsonschema" and channel.message_encoding == "json"
            assert message.log_time == message.publish_time and message.log_time >= previous_time
            previous_time = message.log_time
            if channel.topic == "screen":
                add_screen_reference(
                    messages, source, ScreenPayload.model_validate_json(message.data), message.log_time
                )
            else:
                add_control_event(
                    messages, channel.topic, ActionPayload.model_validate_json(message.data), message.log_time
                )
    if not messages.raw_origins and start_seconds > 0:
        preceding = max(settings.control_state_lookback_seconds, stop_seconds - start_seconds)
        return read_recorded_capture(source, settings, stop_seconds, max(0.0, start_seconds - preceding))
    return finish_capture(messages, settings)
