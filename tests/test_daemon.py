"""
Tests for keyboard discovery and hotkey detection across several keyboards
"""

import os
from unittest.mock import MagicMock

import pytest
from evdev import InputEvent, ecodes

from whisper_dictation import daemon as daemon_mod

KEYBOARD_KEYS = [ecodes.KEY_A, ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTALT]


class FakeDevice:
    """Stands in for evdev.InputDevice; a pipe gives the selector a real fd"""

    def __init__(self, name, keys):
        self.name = name
        self._keys = keys
        self._r, self._w = os.pipe()

    def capabilities(self):
        return {ecodes.EV_KEY: self._keys}

    def fileno(self):
        return self._r

    def close(self):
        for fd in (self._r, self._w):
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.fixture
def system(tmp_path, monkeypatch):
    """A daemon over a mutable set of fake /dev/input nodes"""
    nodes = {}
    monkeypatch.setattr(daemon_mod, "list_devices", lambda: list(nodes))
    monkeypatch.setattr(daemon_mod, "InputDevice", lambda path: nodes[path]())
    for component in ("AudioRecorder", "WhisperTranscriber", "DictationUI", "TextPaster"):
        monkeypatch.setattr(daemon_mod, component, MagicMock())

    config = tmp_path / "config.yaml"
    config.write_text("hotkey:\n  modifiers: [super]\n  key: altgr\n")
    d = daemon_mod.DictationDaemon(config)
    d.stopped = 0

    def fake_stop():
        d.is_recording = False
        d.stopped += 1

    d.stop_recording_and_transcribe = fake_stop

    def plug(path, name, keys=KEYBOARD_KEYS):
        nodes[path] = lambda: FakeDevice(name, keys)

    yield d, nodes, plug
    for path in list(d.devices):
        d._drop_keyboard(path)


def press(d, path, code, value):
    d.on_key_event(InputEvent(0, 0, ecodes.EV_KEY, code, value), path)


def test_reads_every_keyboard_and_skips_non_keyboards(system):
    d, _, plug = system
    plug("/dev/input/event0", "AT Translated Set 2 keyboard")
    plug("/dev/input/event16", "CM Storm Quickfire Rapid i Keyboard")
    plug("/dev/input/event5", "SNSL0028:00 Touchpad", keys=[ecodes.BTN_LEFT])
    plug("/dev/input/event19", "ydotoold virtual device")
    d.scan_keyboards()
    assert set(d.devices) == {"/dev/input/event0", "/dev/input/event16"}


def test_hotkey_on_keyboard_plugged_in_after_start(system):
    d, _, plug = system
    plug("/dev/input/event0", "AT Translated Set 2 keyboard")
    d.scan_keyboards()
    plug("/dev/input/event16", "CM Storm Quickfire Rapid i Keyboard")
    d.scan_keyboards()

    press(d, "/dev/input/event16", ecodes.KEY_LEFTMETA, 1)
    press(d, "/dev/input/event16", ecodes.KEY_RIGHTALT, 1)
    assert d.is_recording
    press(d, "/dev/input/event16", ecodes.KEY_RIGHTALT, 0)
    assert not d.is_recording and d.stopped == 1


def test_chord_split_across_keyboards_does_not_trigger(system):
    d, _, plug = system
    plug("/dev/input/event0", "AT Translated Set 2 keyboard")
    plug("/dev/input/event16", "CM Storm Quickfire Rapid i Keyboard")
    d.scan_keyboards()

    press(d, "/dev/input/event0", ecodes.KEY_LEFTMETA, 1)
    press(d, "/dev/input/event16", ecodes.KEY_RIGHTALT, 1)
    assert not d.is_recording


def test_unplug_while_recording_stops(system):
    d, nodes, plug = system
    plug("/dev/input/event16", "CM Storm Quickfire Rapid i Keyboard")
    d.scan_keyboards()
    press(d, "/dev/input/event16", ecodes.KEY_LEFTMETA, 1)
    press(d, "/dev/input/event16", ecodes.KEY_RIGHTALT, 1)
    assert d.is_recording

    del nodes["/dev/input/event16"]
    d._drop_keyboard("/dev/input/event16")
    assert not d.is_recording and d.stopped == 1
    assert d.devices == {}


def test_reused_node_number_is_probed_again(system):
    d, nodes, plug = system
    plug("/dev/input/event16", "Some Mouse", keys=[ecodes.BTN_LEFT])
    d.scan_keyboards()
    assert d.devices == {}

    del nodes["/dev/input/event16"]
    d.scan_keyboards()
    plug("/dev/input/event16", "CM Storm Quickfire Rapid i Keyboard")
    d.scan_keyboards()
    assert set(d.devices) == {"/dev/input/event16"}


def test_unreadable_node_is_retried(system, monkeypatch):
    d, nodes, plug = system

    def denied():
        raise PermissionError("udev has not set the group yet")

    nodes["/dev/input/event16"] = denied
    d.scan_keyboards()
    assert d.devices == {}
    plug("/dev/input/event16", "CM Storm Quickfire Rapid i Keyboard")
    d.scan_keyboards()
    assert set(d.devices) == {"/dev/input/event16"}
