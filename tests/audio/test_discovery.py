from __future__ import annotations

from pmai_core.resources.audio.discovery import build_fingerprint, microphones_from_devices

HOSTAPIS = [{"name": "ALSA"}]


def _device(name: str, index_rate: int, channels: int = 1) -> dict:
    return {
        "name": name,
        "max_input_channels": channels,
        "hostapi": 0,
        "default_samplerate": float(index_rate),
    }


def test_fingerprint_ignores_sample_rate() -> None:
    devices = [
        _device("USB PnP", 16000),
        _device("USB PnP", 48000),
    ]
    found = microphones_from_devices(devices, HOSTAPIS)
    assert found[0].microphone_id == found[1].microphone_id
    assert found[0].sample_rate == 16000
    assert found[1].sample_rate == 48000
    assert found[0].microphone_id == build_fingerprint("USB PnP", "ALSA", 1)
    assert found[0].hardware_fingerprint == found[0].microphone_id


def test_usb_device_is_listed_when_an_analog_input_is_also_present() -> None:
    devices = [
        _device("Analog", 16000),
        _device("USB Mic", 16000),
    ]
    found = microphones_from_devices(devices, HOSTAPIS)
    assert [item.name for item in found] == ["USB Mic"]


def test_analog_input_is_listed_when_no_usb_input_exists() -> None:
    found = microphones_from_devices([_device("Analog", 16000)], HOSTAPIS)
    assert found[0].name == "Analog"
    assert found[0].device_index == 0


def test_preferred_id_is_listed_first() -> None:
    devices = [
        _device("USB A", 16000),
        _device("USB B", 16000),
    ]
    preferred = build_fingerprint("USB B", "ALSA", 1)
    found = microphones_from_devices(devices, HOSTAPIS, preferred)
    assert found[0].microphone_id == preferred
    assert found[0].device_index == 1
