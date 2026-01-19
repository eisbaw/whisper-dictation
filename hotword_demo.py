#!/usr/bin/env python3
"""
Hotword detection demo using Silero VAD + tiny Whisper.

Listens continuously for a hotword, then activates for full transcription.
Uses minimal resources: VAD for always-on listening, tiny model with 1 thread.
"""

import shutil
import subprocess
import sys
import tempfile
import time
import wave
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


def main():
    print("Loading Silero VAD model...", file=sys.stderr)
    vad_model = load_silero_vad()
    vad_model.eval()

    if not TINY_MODEL.exists():
        print(f"Error: Tiny model not found at {TINY_MODEL}", file=sys.stderr)
        sys.exit(1)

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
        print(f"\n[Checking {duration:.1f}s audio for hotword...]", file=sys.stderr)

        start = time.time()
        text = transcribe(audio_file, TINY_MODEL, threads=1)
        elapsed = time.time() - start

        audio_file.unlink(missing_ok=True)

        if text:
            print(f"[Heard: \"{text}\"] ({elapsed:.2f}s)", file=sys.stderr)
            hotword = check_hotword(text)
            if hotword:
                print(f"\n*** HOTWORD DETECTED: \"{hotword}\" ***", file=sys.stderr)
                print("Listening for command...", file=sys.stderr)
                activated = True
                activation_audio = []
                # Notify
                subprocess.run(
                    ["notify-send", "-t", "2000", "Hotword Detected", f"Listening for command..."],
                    capture_output=True,
                )
        else:
            print(f"[No speech detected] ({elapsed:.2f}s)", file=sys.stderr)

    def process_command(audio_data: np.ndarray):
        """Process command after hotword activation"""
        nonlocal activated

        audio_file = temp_dir / "command.wav"
        save_wav(audio_data, audio_file)

        duration = len(audio_data) / SAMPLE_RATE
        print(f"\n[Transcribing {duration:.1f}s command with base model...]", file=sys.stderr)

        start = time.time()
        text = transcribe(audio_file, BASE_MODEL, threads=4)
        elapsed = time.time() - start

        audio_file.unlink(missing_ok=True)

        if text:
            print(f"\n>>> COMMAND: {text} ({elapsed:.2f}s)", flush=True)
            subprocess.run(
                ["notify-send", "-t", "5000", "Command", text],
                capture_output=True,
            )
        else:
            print(f"[No command detected]", file=sys.stderr)

        activated = False
        print("\nListening for hotword...", file=sys.stderr)

    def audio_callback(indata, frames, time_info, status):
        nonlocal in_speech, silence_chunks, speech_chunks, segment_audio
        nonlocal activated, activation_audio

        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        audio = indata[:, 0].astype(np.float32)
        audio_tensor = torch.from_numpy(audio)

        with torch.no_grad():
            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

        is_speech = speech_prob >= THRESHOLD

        # Visual indicator (minimal)
        if is_speech:
            indicator = "█" * int(speech_prob * 10)
        else:
            indicator = "░"
        mode = "CMD" if activated else "HOT"
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
