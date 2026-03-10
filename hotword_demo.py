#!/usr/bin/env python3
"""
Hotword detection demo using Silero VAD + tiny Whisper.

Listens continuously for a hotword, then activates for full transcription.
Uses minimal resources: VAD for always-on listening, tiny model with 1 thread.

Usage:
    python hotword_demo.py          # Normal mode
    python hotword_demo.py --gui    # Live scrolling ASCII plot
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

# Audio parameters
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # Silero requires exactly 512 samples at 16kHz
CHUNK_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE  # 32ms

# VAD parameters
THRESHOLD = 0.5
MIN_SILENCE_MS = 400  # Shorter silence threshold for hotword
MIN_SPEECH_MS = 200
MAX_SPEECH_MS = 3000  # Max duration for hotword phrase (3 seconds)

MIN_SILENCE_CHUNKS = MIN_SILENCE_MS // CHUNK_MS
MIN_SPEECH_CHUNKS = MIN_SPEECH_MS // CHUNK_MS
MAX_SPEECH_CHUNKS = MAX_SPEECH_MS // CHUNK_MS

# Hotword configuration
HOTWORDS = ["attention computer"]

# Whisper configuration
TINY_MODEL = Path.home() / ".local/share/whisper-models/ggml-tiny.bin"
BASE_MODEL = Path.home() / ".local/share/whisper-models/ggml-base.bin"
WHISPER_LANGUAGE = "en"


def find_whisper_cli():
    """Find whisper-cli binary"""
    whisper = shutil.which("whisper-cli")
    if whisper:
        return whisper
    import glob
    candidates = glob.glob("/nix/store/*whisper-cpp*/bin/whisper-cli")
    if candidates:
        return sorted(candidates)[-1]
    return "whisper-cli"


WHISPER_CLI = find_whisper_cli()


def load_silero_vad():
    """Load Silero VAD model"""
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    return model


def save_wav(audio_data: np.ndarray, filepath: Path):
    """Save audio data to WAV file"""
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())


def transcribe(audio_file: Path, model: Path, threads: int = 1) -> str | None:
    """Transcribe audio file using whisper"""
    try:
        result = subprocess.run(
            [
                WHISPER_CLI,
                "-m", str(model),
                "-f", str(audio_file),
                "--no-timestamps",
                "--language", WHISPER_LANGUAGE,
                "--threads", str(threads),
                "-np",  # No prints except result
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Extract just the transcription text
            text = result.stdout.strip()
            # Filter out blank audio markers
            if text and text not in ["[BLANK_AUDIO]", "(blank audio)"]:
                return text.lower()
        return None
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
        return None


def check_hotword(text: str) -> str | None:
    """Check if text contains a hotword, return the matched hotword"""
    if not text:
        return None
    text_lower = text.lower()
    for hotword in HOTWORDS:
        if hotword in text_lower:
            return hotword
    return None


class AsciiPlot:
    """Live scrolling ASCII plot for VAD visualization"""

    def __init__(self, width: int = 80, height: int = 15):
        self.width = width
        self.height = height
        self.history = deque(maxlen=width)
        self.events = deque(maxlen=width)  # Track events at each position
        self.last_event = ""

    def add(self, value: float, event: str = ""):
        """Add a value (0-1) to the plot"""
        self.history.append(value)
        self.events.append(event)
        if event:
            self.last_event = event

    def render(self, mode: str = "HOT", is_speech: bool = False) -> str:
        """Render the plot as ASCII art"""
        lines = []

        # Header
        status = "▶ SPEECH" if is_speech else "  silence"
        mode_color = "CMD" if mode == "CMD" else "HOT"
        lines.append(f"┌{'─' * (self.width + 2)}┐ [{mode_color}] {status}")

        # Plot area
        for row in range(self.height - 1, -1, -1):
            threshold_row = int(THRESHOLD * (self.height - 1))
            row_char = "│"

            cells = []
            for i, val in enumerate(self.history):
                val_row = int(val * (self.height - 1))
                event = self.events[i] if i < len(self.events) else ""

                if val_row >= row:
                    if event == "hotword":
                        cells.append("█")  # Hotword detected
                    elif event == "command":
                        cells.append("▓")  # Command
                    elif val >= THRESHOLD:
                        cells.append("▓")  # Speech
                    else:
                        cells.append("░")  # Below threshold but has value
                elif row == threshold_row:
                    cells.append("·")  # Threshold line
                else:
                    cells.append(" ")

            # Pad if not enough history
            while len(cells) < self.width:
                cells.insert(0, "·" if int(THRESHOLD * (self.height - 1)) == row else " ")

            # Row label
            if row == self.height - 1:
                label = "1.0"
            elif row == threshold_row:
                label = f"{THRESHOLD:.1f}"
            elif row == 0:
                label = "0.0"
            else:
                label = "   "

            lines.append(f"{row_char}{''.join(cells)}{row_char} {label}")

        # Footer with scale
        lines.append(f"└{'─' * (self.width + 2)}┘")

        # Last event
        if self.last_event:
            lines.append(f"  Last: {self.last_event}")

        return "\n".join(lines)

    def clear_screen(self):
        """Clear screen and move cursor to top"""
        print("\033[2J\033[H", end="")

    def draw(self, mode: str = "HOT", is_speech: bool = False):
        """Clear and redraw the plot"""
        self.clear_screen()
        print(self.render(mode, is_speech))


def main():
    parser = argparse.ArgumentParser(description="Hotword detection demo")
    parser.add_argument("--gui", action="store_true", help="Show live scrolling ASCII plot")
    parser.add_argument("--width", type=int, default=80, help="Plot width (default: 80)")
    parser.add_argument("--height", type=int, default=12, help="Plot height (default: 12)")
    args = parser.parse_args()

    print("Loading Silero VAD model...", file=sys.stderr)
    vad_model = load_silero_vad()
    vad_model.eval()

    if not TINY_MODEL.exists():
        print(f"Error: Tiny model not found at {TINY_MODEL}", file=sys.stderr)
        sys.exit(1)

    # Create plot if GUI mode
    plot = AsciiPlot(width=args.width, height=args.height) if args.gui else None

    if not args.gui:
        print(f"Hotword detection demo", file=sys.stderr)
        print(f"  Hotwords: {', '.join(HOTWORDS)}", file=sys.stderr)
        print(f"  VAD threshold: {THRESHOLD}", file=sys.stderr)
        print(f"  Max hotword duration: {MAX_SPEECH_MS}ms", file=sys.stderr)
        print(f"  Whisper threads: 1 (minimal CPU)", file=sys.stderr)
        print(f"  Tiny model: {TINY_MODEL.name}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Listening for hotword... (Ctrl+C to stop)", file=sys.stderr)
        print("=" * 50, file=sys.stderr)

    # State
    in_speech = False
    silence_chunks = 0
    speech_chunks = 0
    segment_audio = []
    activated = False
    activation_audio = []

    temp_dir = Path(tempfile.gettempdir()) / "hotword-demo"
    temp_dir.mkdir(exist_ok=True)

    def process_hotword_candidate(audio_data: np.ndarray):
        """Process a potential hotword segment"""
        nonlocal activated, activation_audio

        audio_file = temp_dir / "hotword.wav"
        save_wav(audio_data, audio_file)

        duration = len(audio_data) / SAMPLE_RATE
        if not plot:
            print(f"\n[Checking {duration:.1f}s audio for hotword...]", file=sys.stderr)

        start = time.time()
        text = transcribe(audio_file, TINY_MODEL, threads=1)
        elapsed = time.time() - start

        audio_file.unlink(missing_ok=True)

        if text:
            if plot:
                plot.last_event = f'Heard: "{text}" ({elapsed:.2f}s)'
            else:
                print(f"[Heard: \"{text}\"] ({elapsed:.2f}s)", file=sys.stderr)

            hotword = check_hotword(text)
            if hotword:
                if plot:
                    plot.last_event = f'*** HOTWORD: "{hotword}" *** Listening for command...'
                else:
                    print(f"\n*** HOTWORD DETECTED: \"{hotword}\" ***", file=sys.stderr)
                    print("Listening for command...", file=sys.stderr)
                activated = True
                activation_audio = []
                subprocess.run(
                    ["notify-send", "-t", "2000", "Hotword Detected", f"Listening for command..."],
                    capture_output=True,
                )
        else:
            if plot:
                plot.last_event = f"No speech detected ({elapsed:.2f}s)"
            else:
                print(f"[No speech detected] ({elapsed:.2f}s)", file=sys.stderr)

    def process_command(audio_data: np.ndarray):
        """Process command after hotword activation"""
        nonlocal activated

        audio_file = temp_dir / "command.wav"
        save_wav(audio_data, audio_file)

        duration = len(audio_data) / SAMPLE_RATE
        if not plot:
            print(f"\n[Transcribing {duration:.1f}s command with base model...]", file=sys.stderr)

        start = time.time()
        text = transcribe(audio_file, BASE_MODEL, threads=4)
        elapsed = time.time() - start

        audio_file.unlink(missing_ok=True)

        if text:
            if plot:
                plot.last_event = f'COMMAND: "{text}" ({elapsed:.2f}s)'
            else:
                print(f"\n>>> COMMAND: {text} ({elapsed:.2f}s)", flush=True)
            subprocess.run(
                ["notify-send", "-t", "5000", "Command", text],
                capture_output=True,
            )
        else:
            if plot:
                plot.last_event = f"No command detected ({elapsed:.2f}s)"
            else:
                print(f"[No command detected]", file=sys.stderr)

        activated = False
        if not plot:
            print("\nListening for hotword...", file=sys.stderr)

    def audio_callback(indata, frames, time_info, status):
        nonlocal in_speech, silence_chunks, speech_chunks, segment_audio
        nonlocal activated, activation_audio

        if status and not plot:
            print(f"Audio status: {status}", file=sys.stderr)

        audio = indata[:, 0].astype(np.float32)
        audio_tensor = torch.from_numpy(audio)

        with torch.no_grad():
            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

        is_speech = speech_prob >= THRESHOLD
        mode = "CMD" if activated else "HOT"

        # Visual indicator
        if plot:
            plot.add(speech_prob)
            plot.draw(mode=mode, is_speech=is_speech)
        else:
            if is_speech:
                indicator = "█" * int(speech_prob * 10)
            else:
                indicator = "░"
            print(f"\r[{mode}] {indicator:<10} {speech_prob:.2f}  ", end="", file=sys.stderr)

        if is_speech:
            silence_chunks = 0
            speech_chunks += 1

            if activated:
                activation_audio.append(audio.copy())
            else:
                segment_audio.append(audio.copy())
                # Limit hotword segment length
                if speech_chunks > MAX_SPEECH_CHUNKS:
                    # Too long for hotword, reset
                    in_speech = False
                    speech_chunks = 0
                    segment_audio = []

            if not in_speech and speech_chunks >= MIN_SPEECH_CHUNKS:
                in_speech = True
        else:
            if in_speech:
                silence_chunks += 1

                if activated:
                    activation_audio.append(audio.copy())
                else:
                    segment_audio.append(audio.copy())

                if silence_chunks >= MIN_SILENCE_CHUNKS:
                    # Speech ended
                    if activated:
                        # Process as command
                        if activation_audio:
                            audio_data = np.concatenate(activation_audio)
                            process_command(audio_data)
                            activation_audio = []
                    else:
                        # Check for hotword
                        if segment_audio:
                            audio_data = np.concatenate(segment_audio)
                            process_hotword_candidate(audio_data)

                    in_speech = False
                    speech_chunks = 0
                    silence_chunks = 0
                    segment_audio = []
            else:
                speech_chunks = 0
                segment_audio = []

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
