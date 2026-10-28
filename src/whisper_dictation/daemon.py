#!/usr/bin/env python3
"""
Main daemon for whisper-dictation
Monitors keyboard for push-to-talk hotkey and coordinates transcription
"""

import argparse
import logging
import selectors
import signal
import sys
from pathlib import Path

from evdev import InputDevice, ecodes, list_devices

from .config import Config
from .paste import TextPaster
from .recorder import AudioRecorder
from .transcriber import WhisperTranscriber
from .ui import DictationUI

logger = logging.getLogger(__name__)


class DictationDaemon:
    """Main daemon that coordinates all components"""

    def __init__(self, config_path: Path = None):
        self.config = Config(config_path)
        self.recorder = AudioRecorder(self.config)
        self.transcriber = WhisperTranscriber(self.config)
        self.ui = DictationUI(self.config)
        self.paster = TextPaster(self.config)

        # Every keyboard is read, each with its own key state, so a hotkey
        # chord must come from one device. Keyed by /dev/input path.
        self.devices = {}
        self.keys_pressed = {}
        self.not_keyboards = set()
        self.is_recording = False
        self.recording_device = None
        self.selector = selectors.DefaultSelector()

    def _is_virtual_device(self, device_name):
        """Check if device name matches virtual device patterns"""
        virtual_patterns = [
            "virtual",
            "ydotoold",
            "xdotool",
            "uinput",
            "sleep button",
            "power button",
            "lid switch",
            "video bus",
        ]
        return any(pattern.lower() in device_name.lower() for pattern in virtual_patterns)

    def _has_required_keys(self, capabilities):
        """Check if device has letter keys and modifier keys"""
        if ecodes.EV_KEY not in capabilities:
            return False, False
        keys = capabilities[ecodes.EV_KEY]
        has_letters = ecodes.KEY_A in keys
        has_modifiers = (
            ecodes.KEY_LEFTMETA in keys
            or ecodes.KEY_RIGHTMETA in keys
            or ecodes.KEY_LEFTCTRL in keys
        )
        return has_letters, has_modifiers

    def _probe(self, device_path):
        """Open device_path if it is a keyboard; None if not (yet) usable"""
        try:
            device = InputDevice(device_path)
        except (OSError, PermissionError) as e:
            # Not cached: a fresh node may not have its udev permissions yet
            logger.debug(f"Cannot access {device_path}: {e}")
            return None

        has_letters, has_modifiers = self._has_required_keys(device.capabilities())
        reason = None
        if not has_letters:
            reason = "no letter keys"
        elif not has_modifiers:
            reason = "no modifier keys"
        elif self._is_virtual_device(device.name):
            reason = "virtual device"
        if reason:
            logger.debug(f"Skip {device.name} at {device_path}: {reason}")
            self.not_keyboards.add(device_path)
            device.close()
            return None
        return device

    def scan_keyboards(self):
        """Start reading keyboards that appeared since the last scan"""
        present = set(list_devices())
        # Event node numbers are reused, so forget nodes that went away
        self.not_keyboards &= present
        for device_path in sorted(present - self.devices.keys() - self.not_keyboards):
            device = self._probe(device_path)
            if device:
                logger.info(f"Monitoring keyboard: {device.name} at {device_path}")
                self.devices[device_path] = device
                self.keys_pressed[device_path] = set()
                self.selector.register(device, selectors.EVENT_READ, device_path)

    def _drop_keyboard(self, device_path):
        """Stop reading a keyboard that was unplugged"""
        device = self.devices.pop(device_path)
        self.keys_pressed.pop(device_path)
        self.selector.unregister(device)
        try:
            device.close()
        except OSError:
            pass
        logger.info(f"Keyboard gone: {device.name} at {device_path}")
        if self.is_recording and self.recording_device == device_path:
            self.stop_recording_and_transcribe()

    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return

        logger.info("Starting recording...")
        self.is_recording = True

        # Show UI
        self.ui.show_recording()

        # Start recording
        self.recorder.start()

    def stop_recording_and_transcribe(self):
        """Stop recording and transcribe audio"""
        if not self.is_recording:
            return

        logger.info("Stopping recording...")
        self.is_recording = False

        # Stop recorder
        audio_file = self.recorder.stop()

        # Update UI
        self.ui.show_transcribing()

        # Check if we have audio
        if not audio_file or not audio_file.exists() or audio_file.stat().st_size < 10000:
            logger.warning("No audio recorded or file too small")
            self.ui.show_error("No audio recorded")
            return

        # Transcribe in background
        def on_transcription_complete(text):
            if text:
                logger.info(f"Transcription: {text}")
                self.paster.paste(text)
                self.ui.show_success(text)
            else:
                logger.warning("No speech detected")
                self.ui.show_error("No speech detected")

        def on_transcription_error(error):
            logger.error(f"Transcription error: {error}")
            self.ui.show_error(f"Transcription failed: {error}")

        self.transcriber.transcribe_async(
            audio_file, on_complete=on_transcription_complete, on_error=on_transcription_error
        )

    def _track_key_state(self, event, pressed):
        """Track pressed/released keys"""
        if event.value == 1:  # Key down
            pressed.add(event.code)
            logger.debug(f"Key DOWN: {event.code} | Currently pressed: {pressed}")
        elif event.value == 0:  # Key up
            pressed.discard(event.code)
            logger.debug(f"Key UP: {event.code} | Currently pressed: {pressed}")

    def _log_hotkey_debug(self, event, hotkey_key, hotkey_modifiers, has_modifiers, hotkey_pressed):
        """Log hotkey matching for debugging"""
        if event.value == 1 and (event.code == hotkey_key or event.code in hotkey_modifiers):
            logger.info(
                f"Hotkey component pressed: code={event.code}, "
                f"expected_mods={hotkey_modifiers}, expected_key={hotkey_key}, "
                f"has_mods={has_modifiers}, is_hotkey={hotkey_pressed}"
            )

    def on_key_event(self, event, device_path):
        """Handle keyboard events from the keyboard at device_path"""
        if event.type != ecodes.EV_KEY:
            return

        # Track key state
        pressed = self.keys_pressed[device_path]
        self._track_key_state(event, pressed)

        # Get configured hotkey
        hotkey_modifiers = self.config.get_hotkey_modifiers()
        hotkey_key = self.config.get_hotkey_key()

        # Check if ANY of the modifier keys are pressed (e.g., left OR right Super)
        has_modifiers = any(mod in pressed for mod in hotkey_modifiers)
        hotkey_pressed = hotkey_key in pressed

        # Log for debugging
        self._log_hotkey_debug(event, hotkey_key, hotkey_modifiers, has_modifiers, hotkey_pressed)

        # Handle hotkey press
        if (
            has_modifiers
            and event.code == hotkey_key
            and event.value == 1
            and not self.is_recording
        ):
            logger.info("🎤 HOTKEY COMBO DETECTED - Starting recording!")
            self.recording_device = device_path
            self.start_recording()

        # Handle hotkey release
        if (
            event.code == hotkey_key
            and event.value == 0
            and self.is_recording
            and device_path == self.recording_device
        ):
            logger.info("🛑 HOTKEY RELEASED - Stopping recording!")
            self.stop_recording_and_transcribe()

    def run(self):
        """Main daemon loop"""
        logger.info("Starting Whisper Dictation daemon...")

        self.scan_keyboards()
        if not self.devices:
            logger.error("Could not find keyboard device")
            logger.error("Make sure you're in the 'input' group: sudo usermod -aG input $USER")
            sys.exit(1)

        # Setup signal handlers
        def cleanup(sig, frame):
            logger.info("Shutting down...")
            if self.is_recording:
                self.recorder.stop()
            self.ui.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        logger.info(f"Press {self.config.get_hotkey_display()} to start dictation")

        # Initialize UI
        self.ui.show_ready()

        # Event loop
        try:
            while True:
                # The timeout doubles as the hotplug poll for new keyboards
                for key, _ in self.selector.select(timeout=2.0):
                    device_path = key.data
                    try:
                        events = list(key.fileobj.read())
                    except BlockingIOError:
                        continue
                    except OSError:
                        self._drop_keyboard(device_path)
                        continue
                    for event in events:
                        self.on_key_event(event, device_path)
                self.scan_keyboards()
        except KeyboardInterrupt:
            cleanup(None, None)
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            cleanup(None, None)


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="Whisper Dictation Daemon")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging (shows all key events)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging (shows INFO messages)"
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        help="Override language (en, it, es, fr, etc.). Default: from config file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Override model (tiny, base, small, medium, large). Default: from config file",
    )
    args = parser.parse_args()

    # Configure logging based on flags
    log_level = logging.WARNING  # Default: quiet
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    daemon = DictationDaemon()

    # Override config with command-line arguments
    if args.language:
        daemon.config.config["whisper"]["language"] = args.language
        logger.info(f"Language override: {args.language}")

    if args.model:
        daemon.config.config["whisper"]["model"] = args.model
        logger.info(f"Model override: {args.model}")

    daemon.run()


if __name__ == "__main__":
    main()
