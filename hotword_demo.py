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
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
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

# Pre-roll: how much audio to retain BEFORE VAD detects speech onset.
# Silero needs ~32-64 ms above threshold before crossing 0.5, which clips the
# leading consonant of words like "hey". Prepending this buffer recovers it.
PRE_ROLL_MS = 300

# CMD mode timeout: if no speech-start arrives within this long after a wake
# word is detected (and no command-in-same-run was extracted), revert to HOT
# listening. Counted in idle worker iterations (i.e. chunks consumed where no
# run is being built and no transcription is in flight).
CMD_TIMEOUT_MS = 5000

MIN_SILENCE_CHUNKS = MIN_SILENCE_MS // CHUNK_MS
MIN_SPEECH_CHUNKS = MIN_SPEECH_MS // CHUNK_MS
MAX_SPEECH_CHUNKS = MAX_SPEECH_MS // CHUNK_MS
PRE_ROLL_CHUNKS = PRE_ROLL_MS // CHUNK_MS
CMD_TIMEOUT_CHUNKS = CMD_TIMEOUT_MS // CHUNK_MS

# Hotword configuration.
# Matched after canonicalisation (lowercase + punctuation stripped + whitespace
# collapsed) on whole-word boundaries. Tiny model + 1 thread is noisy; if a
# phrase rarely triggers, add common mis-transcriptions here (e.g. "hey cloud",
# "hey clawd") rather than tuning the model.
HOTWORDS = [
    "hey computer",
    "hey claude",
    "hey clawd",  # common whisper-tiny mistranscription of "claude"
]


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def canonicalize(text: str) -> str:
    """Normalise transcribed text for hotword matching.

    Lowercases, replaces all punctuation with whitespace, then collapses
    consecutive whitespace. Result is suitable for whole-word substring checks
    after padding with spaces on both sides.
    """
    if not text:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# Pre-compute canonical forms of the configured hotwords once at import time.
HOTWORDS_CANONICAL = [canonicalize(h) for h in HOTWORDS]


# ---------------------------------------------------------------------------
# "hey claude" -> xterm + claude routing
#
# When the user says one of the CLAUDE_HOTWORDS, the demo spawns (or reuses)
# an xterm window running the `claude` CLI and types the subsequent command
# into it via ydotool. This is the demo equivalent of a voice-controlled
# REPL hand-off.
#
# Caveats (also surfaced at startup):
#   - X11 / Xwayland only (xdotool windowactivate is not supported on
#     native Wayland).
#   - ydotoold must already be running, with the user's $YDOTOOL_SOCKET set
#     or the default path readable.
#   - xterm, xdotool, ydotool, and `claude` must be on $PATH.
# ---------------------------------------------------------------------------
# WM_CLASS resource_name set on the xterm via its `-name` flag. We search by
# this rather than by window title because `claude` CLI overwrites the title
# the moment it starts ("Claude Code", possibly with a project name), so a
# title-based search returns zero hits even though the xterm exists.
CLAUDE_WINDOW_RESNAME = "claude-hotword-demo"
CLAUDE_HOTWORDS = ("hey claude", "hey clawd")


def hotword_targets_claude(hotword: str | None) -> bool:
    """True iff the matched hotword should route into the claude xterm."""
    return hotword in CLAUDE_HOTWORDS


def find_claude_window() -> str | None:
    """Return the X window ID of our claude xterm, or None if not present.

    Matches WM_CLASS resource_name (set by xterm's -name flag) so the search
    survives `claude` rewriting WM_NAME / the icon title on startup.
    """
    try:
        r = subprocess.run(
            ["xdotool", "search", "--classname", CLAUDE_WINDOW_RESNAME],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.split():
        wid = line.strip()
        if wid:
            return wid
    return None


def open_claude_window_async() -> None:
    """Spawn xterm running `claude` if no window with our title exists.

    Returns immediately; the spawn runs in a daemon thread so the audio
    worker is never blocked.
    """

    def _do():
        existing = find_claude_window()
        if existing:
            print(f"[claude xterm already open wid={existing}]", file=sys.stderr)
            return
        display = os.environ.get("DISPLAY", "<unset>")
        print(
            f"[spawning xterm -name {CLAUDE_WINDOW_RESNAME} -e claude  " f"DISPLAY={display}]",
            file=sys.stderr,
        )
        try:
            proc = subprocess.Popen(
                [
                    "xterm",
                    "-T",
                    CLAUDE_WINDOW_RESNAME,
                    "-name",
                    CLAUDE_WINDOW_RESNAME,
                    "-geometry",
                    "120x36",
                    "-fa",
                    "Monospace",
                    "-fs",
                    "11",
                    "-e",
                    "claude",
                ],
                # Capture stderr so we see X errors (e.g. "Can't open
                # display"). The process is long-running on success; we read
                # stderr only if it exits within 1 s.
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[xterm spawn raised: {type(e).__name__}: {e}]", file=sys.stderr)
            return
        # If xterm exited within 1 s, something failed -- surface its stderr.
        try:
            rc = proc.wait(timeout=1.0)
            err = (
                proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
            )
            print(
                f"[xterm exited fast: rc={rc}  stderr={err!r}  DISPLAY={display}]",
                file=sys.stderr,
            )
        except subprocess.TimeoutExpired:
            # Good -- xterm is still alive after 1 s, meaning the window came up.
            pass

    threading.Thread(target=_do, daemon=True).start()


def _run_logged(argv, *, timeout: float, env: dict | None = None) -> bool:
    """subprocess.run with stdout/stderr captured AND surfaced on error.

    Returns True iff exit was 0. Used for ydotool / xdotool invocations so
    silent failures (wrong socket, permission denied, missing window) are
    visible in the demo log instead of being swallowed.
    """
    try:
        r = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as e:
        print(f"[exec failed: {e}]", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"[exec timeout: {' '.join(argv)}]", file=sys.stderr)
        return False
    if r.returncode != 0:
        print(
            f"[{argv[0]} exit={r.returncode}  "
            f"stderr={r.stderr.strip()!r}  stdout={r.stdout.strip()!r}]",
            file=sys.stderr,
        )
        return False
    return True


def send_to_claude_async(text: str) -> None:
    """Focus the claude xterm and type ``text`` followed by Enter.

    Waits up to ~5 s for the window to appear (in case it was just spawned).
    Uses xdotool to activate the window and ydotool to type. Runs in a
    daemon thread so the audio worker is never blocked.

    The YDOTOOL_SOCKET env var is forwarded explicitly. If it isn't set in
    the parent process, we fall back to the NixOS-service default at
    /run/ydotoold/socket so the demo works out of the box on this machine.
    """
    if not text:
        return

    def _do():
        print(f"[send_to_claude_async entered: {text!r}]", file=sys.stderr)
        wid = find_claude_window()
        if not wid:
            for _ in range(25):  # up to ~5 s for the spawned xterm to appear
                time.sleep(0.2)
                wid = find_claude_window()
                if wid:
                    break

        # Explicit env so the subprocess gets YDOTOOL_SOCKET even if the
        # parent shell didn't propagate it through nix-shell.
        env = dict(os.environ)
        env.setdefault("YDOTOOL_SOCKET", "/run/ydotoold/socket")

        if wid:
            print(f"[routing to claude xterm wid={wid}: {text!r}]", file=sys.stderr)
            if not _run_logged(["xdotool", "windowactivate", "--sync", wid], timeout=3):
                # Activation failed -- fall through to global typing rather
                # than abort. ydotool will hit whatever is focused.
                print(
                    "[xdotool activate failed; falling back to global type]",
                    file=sys.stderr,
                )
            # Let the WM finish focusing and let xterm's input field be ready
            # before we start typing.
            time.sleep(0.5)
        else:
            # Window resolution failed. ydotool types into whatever has focus.
            # If the user just spoke a wake word, the spawned xterm should be
            # at or near the front. Worst case: typed into the wrong window.
            print(
                f"[claude window not resolved; " f"typing globally into focused window: {text!r}]",
                file=sys.stderr,
            )

        if not _run_logged(["ydotool", "type", text], timeout=15, env=env):
            return
        time.sleep(0.05)
        # KEY_ENTER = 28 on Linux; press then release.
        _run_logged(["ydotool", "key", "28:1", "28:0"], timeout=3, env=env)

    threading.Thread(target=_do, daemon=True).start()


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
                "-m",
                str(model),
                "-f",
                str(audio_file),
                "--no-timestamps",
                "--language",
                WHISPER_LANGUAGE,
                "--threads",
                str(threads),
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
    """Check if canonicalised text contains a hotword on word boundaries.

    Returns the original (non-canonical) hotword on match so logs/notifications
    show the human-readable form. Equivalent to ``extract_hotword_and_tail()[0]``
    for callers that don't care about the trailing text.
    """
    hotword, _ = extract_hotword_and_tail(text)
    return hotword


def extract_hotword_and_tail(text: str) -> tuple[str | None, str]:
    """Find a hotword in ``text`` and return ``(hotword, tail)``.

    ``tail`` is the canonicalised text following the matched hotword phrase,
    stripped of surrounding whitespace. It is empty when the hotword was
    spoken in isolation (no command in the same run). When no hotword matches
    the return is ``(None, "")``.

    The tail comes from the (canonicalised) tiny-model transcription, so its
    accuracy is bounded by what whisper-tiny produced. For higher quality you
    would re-transcribe the same audio with a larger model -- not done here.
    """
    if not text:
        return None, ""
    canon = canonicalize(text)
    if not canon:
        return None, ""
    # Pad with spaces so " hey computer " can only match whole words, not
    # the "hey computer" inside "they computer".
    canon_padded = f" {canon} "
    for hotword, hotword_canon in zip(HOTWORDS, HOTWORDS_CANONICAL, strict=True):
        needle = f" {hotword_canon} "
        idx = canon_padded.find(needle)
        if idx >= 0:
            tail = canon_padded[idx + len(needle) :].strip()
            return hotword, tail
    return None, ""


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
        print("Hotword detection demo", file=sys.stderr)
        print(f"  Hotwords: {', '.join(HOTWORDS)}", file=sys.stderr)
        print(f"  VAD threshold: {THRESHOLD}", file=sys.stderr)
        print(f"  Min speech / silence: {MIN_SPEECH_MS}ms / {MIN_SILENCE_MS}ms", file=sys.stderr)
        print(f"  Pre-roll: {PRE_ROLL_MS}ms", file=sys.stderr)
        print("  Whisper threads: 1 (minimal CPU)", file=sys.stderr)
        print(f"  Tiny model: {TINY_MODEL.name}", file=sys.stderr)
        # Tool availability for "hey claude" -> xterm + claude routing
        have_xterm = shutil.which("xterm") is not None
        have_xdotool = shutil.which("xdotool") is not None
        have_ydotool = shutil.which("ydotool") is not None
        have_claude = shutil.which("claude") is not None
        ydotoold_running = (
            subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True).returncode == 0
        )
        print("  Claude routing:", file=sys.stderr)
        print(
            f"    xterm={have_xterm} xdotool={have_xdotool} "
            f"ydotool={have_ydotool} ydotoold={ydotoold_running} "
            f"claude={have_claude}",
            file=sys.stderr,
        )
        if not all([have_xterm, have_xdotool, have_ydotool, have_claude]):
            print(
                "    -> missing tools; 'hey claude' will detect but won't route.",
                file=sys.stderr,
            )
        if not ydotoold_running:
            print(
                "    -> ydotoold not running; typing will fail. "
                "Start it with: ydotoold --socket-path=/run/user/$(id -u)/.ydotool_socket "
                "--socket-perm=0600 &",
                file=sys.stderr,
            )
        print("", file=sys.stderr)
        print("Listening for hotword... (Ctrl+C to stop)", file=sys.stderr)
        print("=" * 50, file=sys.stderr)

    temp_dir = Path(tempfile.gettempdir()) / "hotword-demo"
    temp_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Threaded architecture
    #
    # audio_callback (PortAudio RT thread)  --> audio_q  (raw chunks only)
    #                                                  │
    #                                                  ▼
    #                                       vad_worker (single thread)
    #                                       ├─ run Silero VAD per chunk
    #                                       ├─ maintain pre-roll ring
    #                                       ├─ group contiguous above-threshold
    #                                       │  chunks (with leading pre-roll +
    #                                       │  trailing silence pad) into "runs"
    #                                       ├─ on run-close: spawn a transcription
    #                                       │  thread (concatenated audio).
    #                                       │  Worker KEEPS CONSUMING so audio
    #                                       │  arriving during transcription is
    #                                       │  never lost — the next run can be
    #                                       │  the command following a wake word.
    #                                       │
    #                                       └─ drain result_q every iteration:
    #                                          hotword-check result + match  →
    #                                            next run is the command
    #                                          hotword-check result + no match →
    #                                            discard the audio (text only is
    #                                            kept for the score-trace log)
    #                                          command result → notify, back to HOT
    #
    # Transcription runs in short-lived daemon threads. At most one is in flight
    # at a time (the worker queues additional completed runs).
    # ------------------------------------------------------------------

    audio_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    pending_runs: deque = deque()  # closed runs waiting for the in-flight tx to finish
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        # Real-time PortAudio callback: copy and queue, nothing else.
        if status and not plot:
            print(f"Audio status: {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].astype(np.float32).copy())

    def save_and_transcribe(audio_data: np.ndarray, model_path: Path, threads: int, purpose: str):
        """Run whisper in a worker thread; post (purpose, text, elapsed) to result_q."""
        audio_file = temp_dir / f"{purpose}-{int(time.time() * 1000)}.wav"
        save_wav(audio_data, audio_file)
        start = time.time()
        text = transcribe(audio_file, model_path, threads=threads)
        elapsed = time.time() - start
        audio_file.unlink(missing_ok=True)
        result_q.put((purpose, text, elapsed))

    def vad_worker():
        """Consume audio chunks, group into runs by VAD, dispatch transcriptions."""
        # Pre-roll: rolling window of the last PRE_ROLL_MS, always populated.
        # Stored as (chunk, score) so the score annotation is also retained when
        # the pre-roll seeds a new run.
        pre_roll: deque = deque(maxlen=PRE_ROLL_CHUNKS)

        # Current run being built. Stored as (chunk, score) for per-chunk
        # annotation. A run starts when MIN_SPEECH chunks above threshold have
        # accumulated and closes when MIN_SILENCE chunks below threshold follow.
        current_run: list = []
        speech_chunks = 0  # consecutive above-threshold chunks
        silence_chunks = 0  # consecutive below-threshold chunks during a run
        in_run = False  # passed MIN_SPEECH gate

        # Higher-level state
        activated = False  # True between hotword detection and CMD result
        tx_in_flight = False  # True while a whisper subprocess thread is alive
        # Idle chunks since the most recent speech-burst ended (or since CMD
        # mode was entered with no speech at all). Each burst resets this to 0.
        # When it crosses CMD_TIMEOUT_CHUNKS the accumulated command is
        # dispatched. Drives both the visible countdown and the timeout firing.
        cmd_idle_chunks = 0
        # In CMD mode, individual VAD-delimited speech bursts are NOT dispatched
        # as separate commands. They are appended to this accumulator and the
        # whole thing is dispatched as a single command when CMD_TIMEOUT_MS of
        # silence elapses with no further speech. This lets the user pause
        # naturally mid-command without each pause closing CMD mode.
        command_accumulator: list = []  # list of (chunk, score) tuples
        n_bursts_accumulated = 0
        # The hotword that opened the current CMD-mode session, retained so
        # the post-CMD command result can be routed to the right destination
        # (currently: claude xterm for "hey claude" / "hey clawd", notify-send
        # otherwise). Cleared when CMD ends or on timeout-without-speech.
        cmd_hotword: str | None = None

        def dispatch_run(run, purpose):
            """Concatenate a run's chunks and spawn a transcription thread."""
            nonlocal tx_in_flight
            audio_data = np.concatenate([c for c, _ in run])
            # Score stats over the speech portion of the run (above threshold)
            speech_scores = [s for _, s in run if s >= THRESHOLD]
            avg_score = sum(speech_scores) / len(speech_scores) if speech_scores else 0.0
            duration = len(audio_data) / SAMPLE_RATE
            n_speech = len(speech_scores)
            n_total = len(run)
            label = "command" if purpose == "command" else "hotword-check"
            msg = (
                f"[Run -> {label}]  dur={duration:.2f}s  "
                f"chunks={n_speech}/{n_total} (speech/total)  "
                f"avg_speech_score={avg_score:.2f}"
            )
            if plot:
                plot.last_event = msg
            else:
                print(f"\n{msg}", file=sys.stderr)

            model = BASE_MODEL if purpose == "command" else TINY_MODEL
            threads = 4 if purpose == "command" else 1
            tx_in_flight = True
            threading.Thread(
                target=save_and_transcribe,
                args=(audio_data, model, threads, purpose),
                daemon=True,
            ).start()

        def maybe_dispatch_next():
            """If no transcription is in flight, dispatch the next pending run."""
            if tx_in_flight or not pending_runs:
                return
            run = pending_runs.popleft()
            purpose = "command" if activated else "hotword-check"
            dispatch_run(run, purpose)

        def handle_result(purpose, text, elapsed):
            nonlocal activated, tx_in_flight, cmd_idle_chunks
            nonlocal n_bursts_accumulated, cmd_hotword
            tx_in_flight = False

            if not text:
                msg = f"[{purpose}: <no speech>] ({elapsed:.2f}s)"
                if plot:
                    plot.last_event = msg
                else:
                    print(f"\n{msg}", file=sys.stderr)
                return

            if purpose == "hotword-check":
                hotword, tail = extract_hotword_and_tail(text)
                if hotword:
                    bar = "=" * 60
                    detect_msg = (
                        f'*** HOTWORD: "{hotword}" *** ' f'(heard: "{text}", {elapsed:.2f}s)'
                    )
                    if plot:
                        plot.last_event = detect_msg
                    else:
                        print(f"\n\n{bar}\n{detect_msg}", file=sys.stderr)
                    subprocess.run(
                        ["notify-send", "-t", "2000", "Hotword Detected", hotword],
                        capture_output=True,
                    )
                    # If this wake word routes to claude, kick off the xterm
                    # spawn now (in the background) so the window is ready by
                    # the time we have a command to send.
                    if hotword_targets_claude(hotword):
                        open_claude_window_async()
                    # If the wake word and the command were spoken in one breath
                    # (no pause), the command text is already inside this run's
                    # transcription as "tail". Emit it immediately and DO NOT
                    # activate -- the next run is unrelated.
                    if tail:
                        cmd_msg = f'>>> COMMAND (same run, tiny-model): "{tail}"'
                        if plot:
                            plot.last_event = cmd_msg
                        else:
                            print(f"{cmd_msg}\n{bar}\n", flush=True)
                        if hotword_targets_claude(hotword):
                            send_to_claude_async(tail)
                        else:
                            subprocess.run(
                                ["notify-send", "-t", "5000", "Command", tail],
                                capture_output=True,
                            )
                    else:
                        # Wake word spoken in isolation: enter CMD mode. The
                        # next bursts accumulate into one command; final
                        # dispatch happens on CMD_TIMEOUT_MS of silence.
                        cmd_idle_chunks = 0
                        command_accumulator.clear()
                        n_bursts_accumulated = 0
                        cmd_hotword = hotword
                        banner = (
                            f">>> CMD MODE: listening for command "
                            f"(timeout in {CMD_TIMEOUT_MS // 1000}s, "
                            f"each burst resets the window)"
                        )
                        if plot:
                            plot.last_event = banner
                        else:
                            print(f"{banner}\n{bar}\n", file=sys.stderr)
                        subprocess.run(
                            [
                                "notify-send",
                                "-t",
                                str(CMD_TIMEOUT_MS),
                                "Command Mode",
                                f"Listening (timeout in {CMD_TIMEOUT_MS // 1000}s)",
                            ],
                            capture_output=True,
                        )
                        activated = True
                else:
                    # No match -- audio already discarded after dispatch; just trace.
                    msg = f'[heard, no hotword: "{text}"] ({elapsed:.2f}s)'
                    if plot:
                        plot.last_event = msg
                    else:
                        print(f"\n{msg}", file=sys.stderr)
            elif purpose == "command":
                bar = "=" * 60
                msg = f'>>> COMMAND: "{text}" ({elapsed:.2f}s)'
                if plot:
                    plot.last_event = msg
                else:
                    print(f"\n\n{bar}\n{msg}\n{bar}\n", flush=True)
                if hotword_targets_claude(cmd_hotword):
                    send_to_claude_async(text)
                else:
                    subprocess.run(
                        ["notify-send", "-t", "5000", "Command", text],
                        capture_output=True,
                    )
                activated = False
                cmd_idle_chunks = 0
                command_accumulator.clear()
                n_bursts_accumulated = 0
                cmd_hotword = None

        # ------------------------------------------------------------------
        # Main worker loop
        # ------------------------------------------------------------------
        while not stop_event.is_set():
            # Drain any completed transcriptions FIRST so state (activated /
            # tx_in_flight) is current before we decide what to do with the
            # next pending run.
            while True:
                try:
                    purpose, text, elapsed = result_q.get_nowait()
                except queue.Empty:
                    break
                handle_result(purpose, text, elapsed)
            maybe_dispatch_next()

            try:
                chunk = audio_q.get(timeout=0.05)
            except queue.Empty:
                continue

            audio_tensor = torch.from_numpy(chunk)
            with torch.no_grad():
                score = vad_model(audio_tensor, SAMPLE_RATE).item()

            pre_roll.append((chunk, score))
            is_speech = score >= THRESHOLD
            mode = "CMD" if activated else "HOT"

            # CMD timeout: count chunks while in CMD mode with no speech being
            # captured and no transcription in flight. Each speech burst resets
            # the counter. When the counter exceeds the budget:
            #   - if bursts have accumulated: that IS the end-of-command signal.
            #     Dispatch the accumulated audio as one command (base model).
            #   - if no bursts at all: the user said the wake word and went
            #     silent. Fire a "true timeout" banner and return to HOT.
            if activated and not current_run and not tx_in_flight:
                cmd_idle_chunks += 1
                if cmd_idle_chunks >= CMD_TIMEOUT_CHUNKS:
                    bar = "=" * 60
                    if command_accumulator:
                        total_dur = len(command_accumulator) * CHUNK_MS / 1000.0
                        eoc_msg = (
                            f">>> END OF COMMAND  "
                            f"({CMD_TIMEOUT_MS // 1000}s silence, "
                            f"{n_bursts_accumulated} burst(s), "
                            f"{total_dur:.2f}s total)  -- transcribing..."
                        )
                        if plot:
                            plot.last_event = eoc_msg
                        else:
                            print(f"\n\n{bar}\n{eoc_msg}\n{bar}\n", file=sys.stderr)
                        # Dispatch the accumulator directly as a command. We're
                        # already in the worker thread; dispatch_run just spawns
                        # the whisper subprocess in another thread.
                        accumulated = list(command_accumulator)
                        command_accumulator.clear()
                        n_bursts_accumulated = 0
                        cmd_idle_chunks = 0
                        # NOTE: leave `activated` True until the command result
                        # comes back. handle_result() will set it to False then.
                        # While transcription is in flight, tx_in_flight is True
                        # so this branch won't re-fire, and any new bursts that
                        # arrive will accumulate into the (now-cleared) buffer.
                        # They would be lost when the result clears state. The
                        # user is expected to wait for the command result.
                        dispatch_run(accumulated, "command")
                    else:
                        timeout_msg = (
                            f"... CMD timeout ({CMD_TIMEOUT_MS // 1000}s silence, "
                            f"no speech). Returning to hotword listening."
                        )
                        if plot:
                            plot.last_event = timeout_msg
                        else:
                            print(f"\n\n{bar}\n{timeout_msg}\n{bar}\n", file=sys.stderr)
                        subprocess.run(
                            [
                                "notify-send",
                                "-t",
                                "3000",
                                "CMD Timeout",
                                "Returning to hotword mode",
                            ],
                            capture_output=True,
                        )
                        activated = False
                        cmd_idle_chunks = 0
                        cmd_hotword = None
            else:
                cmd_idle_chunks = 0

            # Per-chunk annotation: every chunk shows its VAD score.
            # When we're in CMD-listening (activated, no run yet, no transcription
            # in flight), also surface the live countdown of the silence budget
            # so the user can see the timeout approaching.
            if plot:
                plot.add(score)
                plot.draw(mode=mode, is_speech=is_speech)
            else:
                indicator = "█" * int(score * 10) if is_speech else "░"
                cmd_listening = activated and not current_run and not tx_in_flight
                # \033[K clears from the cursor to end of line, so a shorter
                # line on the next chunk doesn't leave garbage from the longer
                # CMD-LISTENING countdown text behind it.
                if cmd_listening:
                    remaining_chunks = max(0, CMD_TIMEOUT_CHUNKS - cmd_idle_chunks)
                    remaining_s = remaining_chunks * CHUNK_MS / 1000.0
                    if n_bursts_accumulated > 0:
                        total_dur = len(command_accumulator) * CHUNK_MS / 1000.0
                        accum = (
                            f"  ({n_bursts_accumulated} burst(s), " f"{total_dur:.2f}s captured)"
                        )
                    else:
                        accum = ""
                    print(
                        f"\r\033[K[CMD] LISTENING  {remaining_s:4.1f}s left"
                        f"{accum}  {indicator:<10} {score:.2f}",
                        end="",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"\r\033[K[{mode}] {indicator:<10} {score:.2f}",
                        end="",
                        file=sys.stderr,
                    )

            if is_speech:
                silence_chunks = 0
                speech_chunks += 1
                if not current_run:
                    # First above-threshold chunk after silence: seed the run
                    # with the pre-roll window so the leading consonant of the
                    # word is not lost to VAD onset latency. pre_roll already
                    # contains the current chunk as its last element.
                    current_run = list(pre_roll)
                else:
                    current_run.append((chunk, score))
                if not in_run and speech_chunks >= MIN_SPEECH_CHUNKS:
                    in_run = True
            else:
                if in_run:
                    # Trailing silence is included so whisper sees a clean end.
                    current_run.append((chunk, score))
                    silence_chunks += 1
                    if silence_chunks >= MIN_SILENCE_CHUNKS:
                        if activated:
                            # CMD mode: accumulate the burst into the command
                            # buffer rather than dispatching it. Final dispatch
                            # happens when CMD_TIMEOUT elapses with no further
                            # bursts. Reset the idle counter so the user has
                            # the full timeout window starting from now.
                            command_accumulator.extend(current_run)
                            n_bursts_accumulated += 1
                            burst_dur = len(current_run) * CHUNK_MS / 1000.0
                            total_dur = len(command_accumulator) * CHUNK_MS / 1000.0
                            note = (
                                f"  [CMD burst #{n_bursts_accumulated} captured "
                                f"({burst_dur:.2f}s)  accumulator total: "
                                f"{total_dur:.2f}s]"
                            )
                            if plot:
                                plot.last_event = note.strip()
                            else:
                                print(f"\n{note}", file=sys.stderr)
                            cmd_idle_chunks = 0
                        else:
                            # HOT mode: dispatch the run for hotword-check.
                            pending_runs.append(current_run)
                            maybe_dispatch_next()
                        current_run = []
                        speech_chunks = 0
                        silence_chunks = 0
                        in_run = False
                else:
                    # Pre-MIN_SPEECH blip -- drop it without dispatching.
                    if current_run:
                        current_run = []
                    speech_chunks = 0

    worker = threading.Thread(target=vad_worker, daemon=True)
    worker.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped.", file=sys.stderr)
    finally:
        stop_event.set()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
