#!/usr/bin/env python3
"""
Real-time VAD demo with automatic transcription.
Records from microphone, detects speech segments, transcribes them,
and shows desktop notifications with the results.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch


def find_whisper_cli():
    """Find whisper-cli binary"""
    # Check PATH first
    whisper = shutil.which("whisper-cli")
    if whisper:
        return whisper
    # Check common nix store locations
    import glob
    candidates = glob.glob("/nix/store/*whisper-cpp*/bin/whisper-cli")
    if candidates:
        # Return the newest version
        return sorted(candidates)[-1]
    return "whisper-cli"  # Fall back, will fail if not found


WHISPER_CLI = find_whisper_cli()

# Audio parameters
SAMPLE_RATE = 16000  # Silero VAD requires 16kHz
CHUNK_SAMPLES = 512  # Silero requires exactly 512 samples at 16kHz
CHUNK_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE  # 32ms

# VAD parameters
THRESHOLD = 0.5  # Speech probability threshold
MIN_SILENCE_MS = 500  # Minimum silence to end a segment
MIN_SPEECH_MS = 250  # Minimum speech duration to count as segment

MIN_SILENCE_CHUNKS = MIN_SILENCE_MS // CHUNK_MS
MIN_SPEECH_CHUNKS = MIN_SPEECH_MS // CHUNK_MS

# Whisper parameters
WHISPER_MODEL = Path.home() / ".local/share/whisper-models/ggml-base.bin"
WHISPER_LANGUAGE = "en"


def load_silero_vad():
    """Load Silero VAD model from torch hub"""
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
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())


def transcribe_audio(audio_file: Path, segment_num: int):
    """Transcribe audio file using WHISPER_CLI and show notification"""
    try:
        # Check audio file
        if not audio_file.exists():
            print(f"\n[ERROR] Audio file not found: {audio_file}", file=sys.stderr)
            return

        file_size = audio_file.stat().st_size
        print(f"\n[Transcribing segment {segment_num}] File: {audio_file} ({file_size} bytes)", file=sys.stderr)

        output_file = audio_file.with_suffix("")

        cmd = [
            WHISPER_CLI,
            "-m", str(WHISPER_MODEL),
            "-f", str(audio_file),
            "--output-txt",
            "--output-file", str(output_file),
            "--no-timestamps",
            "--language", WHISPER_LANGUAGE,
            "--threads", "4",
        ]

        print(f"[CMD] {' '.join(cmd)}", file=sys.stderr)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"\n[ERROR] Whisper failed (code {result.returncode}):", file=sys.stderr)
            print(f"  stdout: {result.stdout}", file=sys.stderr)
            print(f"  stderr: {result.stderr}", file=sys.stderr)
            return

        # Read transcription
        txt_file = output_file.with_suffix(".txt")
        print(f"[Looking for] {txt_file}", file=sys.stderr)

        if txt_file.exists():
            text = txt_file.read_text().strip()
            print(f"[Raw text] '{text}'", file=sys.stderr)
            txt_file.unlink()

            if text and text not in ["[BLANK_AUDIO]", "(blank audio)"]:
                # Show notification
                notify_result = subprocess.run(
                    ["notify-send", "-t", "5000", f"Transcription #{segment_num}", text],
                    capture_output=True,
                    text=True,
                )
                if notify_result.returncode != 0:
                    print(f"[WARN] notify-send failed: {notify_result.stderr}", file=sys.stderr)

                # Also print to stdout
                print(f"\n>>> [Segment {segment_num}] {text}", flush=True)
            else:
                print(f"\n[Segment {segment_num}] (blank audio)", file=sys.stderr)
        else:
            print(f"[ERROR] Transcription file not created: {txt_file}", file=sys.stderr)

        # Cleanup audio file
        audio_file.unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        print(f"\n[ERROR] Transcription timeout for segment {segment_num}", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Transcription error: {e}", file=sys.stderr)
        traceback.print_exc()


def main():
    print("Loading Silero VAD model...", file=sys.stderr)
    model = load_silero_vad()
    model.eval()

    if not WHISPER_MODEL.exists():
        print(f"Error: Whisper model not found at {WHISPER_MODEL}", file=sys.stderr)
        sys.exit(1)

    print(f"VAD parameters:", file=sys.stderr)
    print(f"  Sample rate: {SAMPLE_RATE} Hz", file=sys.stderr)
    print(f"  Chunk size: {CHUNK_MS} ms ({CHUNK_SAMPLES} samples)", file=sys.stderr)
    print(f"  Threshold: {THRESHOLD}", file=sys.stderr)
    print(f"  Min silence: {MIN_SILENCE_MS} ms", file=sys.stderr)
    print(f"  Min speech: {MIN_SPEECH_MS} ms", file=sys.stderr)
    print(f"  Whisper model: {WHISPER_MODEL.name}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Listening... (Ctrl+C to stop)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # State tracking
    in_speech = False
    speech_start_time = None
    silence_chunks = 0
    speech_chunks = 0
    segment_count = 0

    # Audio buffer for the current segment
    segment_audio = []

    # Temp directory for audio files
    temp_dir = Path(tempfile.gettempdir()) / "vad-demo"
    temp_dir.mkdir(exist_ok=True)

    def audio_callback(indata, frames, time_info, status):
        nonlocal in_speech, speech_start_time, silence_chunks, speech_chunks
        nonlocal segment_count, segment_audio

        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        # Convert to mono float32 tensor
        audio = indata[:, 0].astype(np.float32)
        audio_tensor = torch.from_numpy(audio)

        # Get speech probability
        with torch.no_grad():
            speech_prob = model(audio_tensor, SAMPLE_RATE).item()

        is_speech = speech_prob >= THRESHOLD
        current_time = time.time()

        # Visual indicator
        bar_len = int(speech_prob * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        state = "SPEECH" if is_speech else "silence"
        indicator = ">>>" if in_speech else "   "
        print(f"\r{indicator} [{bar}] {speech_prob:.2f} {state}   ", end="", file=sys.stderr)

        if is_speech:
            silence_chunks = 0
            speech_chunks += 1
            segment_audio.append(audio.copy())

            if not in_speech and speech_chunks >= MIN_SPEECH_CHUNKS:
                # Speech started
                in_speech = True
                speech_start_time = current_time - (speech_chunks * CHUNK_MS / 1000)
                print(f"\n>>> Speech started", file=sys.stderr)
        else:
            if in_speech:
                silence_chunks += 1
                segment_audio.append(audio.copy())

                if silence_chunks >= MIN_SILENCE_CHUNKS:
                    # Speech ended
                    duration = current_time - speech_start_time - (MIN_SILENCE_MS / 1000)
                    segment_count += 1

                    print(f"\n<<< Speech ended: {duration:.2f}s - transcribing...", file=sys.stderr)

                    # Save audio and transcribe in background thread
                    if segment_audio:
                        audio_data = np.concatenate(segment_audio)
                        audio_file = temp_dir / f"segment_{segment_count}.wav"
                        save_wav(audio_data, audio_file)

                        # Start transcription in background thread
                        thread = threading.Thread(
                            target=transcribe_audio,
                            args=(audio_file, segment_count),
                            daemon=True,
                        )
                        thread.start()

                    # Reset state
                    in_speech = False
                    speech_start_time = None
                    silence_chunks = 0
                    speech_chunks = 0
                    segment_audio = []
            else:
                speech_chunks = 0
                segment_audio = []

    # Start audio stream
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
        print(f"Total segments detected: {segment_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
