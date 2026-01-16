# Whisper Dictation development commands

# Default recipe - show available commands
default:
    @just --list

# Run the daemon with default settings
run:
    python -m whisper_dictation.daemon

# Run daemon with verbose output
run-verbose:
    python -m whisper_dictation.daemon --verbose

# Run daemon with auto language detection
run-auto:
    python -m whisper_dictation.daemon --language auto

# Run daemon with English
run-en:
    python -m whisper_dictation.daemon --language en

# Run daemon with base model (faster)
run-fast:
    python -m whisper_dictation.daemon --model base

# Run tests
test:
    pytest

# Run tests with coverage
test-cov:
    pytest --cov=whisper_dictation --cov-report=term-missing

# Format code
format:
    black src/ tests/
    ruff check --fix src/ tests/

# Lint code
lint:
    ruff check src/ tests/
    black --check src/ tests/

# Run all quality checks
check: lint test

# Test recording and transcription (5 seconds)
test-dictation:
    python -c '\
    import time; \
    from whisper_dictation.config import Config; \
    from whisper_dictation.recorder import AudioRecorder; \
    from whisper_dictation.transcriber import WhisperTranscriber; \
    config = Config(); \
    config.config["whisper"]["model"] = "base"; \
    recorder = AudioRecorder(config); \
    transcriber = WhisperTranscriber(config); \
    print("Get ready..."); \
    time.sleep(2); \
    print(">>> RECORDING NOW - SPEAK! <<<"); \
    recorder.start(); \
    time.sleep(5); \
    audio_file = recorder.stop(); \
    print(">>> STOPPED <<<"); \
    print("Transcribing..."); \
    text = transcriber.transcribe(audio_file); \
    print(f"Result: {text}"); \
    '

# Download whisper base model (~142MB)
download-model-base:
    mkdir -p ~/.local/share/whisper-models
    curl -L -o ~/.local/share/whisper-models/ggml-base.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# Download whisper medium model (~1.5GB)
download-model-medium:
    mkdir -p ~/.local/share/whisper-models
    curl -L -o ~/.local/share/whisper-models/ggml-medium.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin

# Start ydotool daemon (required for text pasting)
start-ydotool:
    ydotoold --socket-path=/run/user/$(id -u)/.ydotool_socket --socket-perm=0600 &

# Check if user is in input group
check-input-group:
    @groups | grep -q input && echo "OK: User is in input group" || echo "MISSING: Add yourself with: sudo usermod -aG input $USER"

# Build nix package
build:
    nix build

# Show setup status
status:
    @echo "=== Whisper Dictation Setup Status ==="
    @echo ""
    @echo "Input group:"
    @groups | grep -q input && echo "  OK: User is in input group" || echo "  MISSING: sudo usermod -aG input $USER (then logout/login)"
    @echo ""
    @echo "Whisper models:"
    @ls -lh ~/.local/share/whisper-models/*.bin 2>/dev/null || echo "  MISSING: Run 'just download-model-base'"
    @echo ""
    @echo "ydotool daemon:"
    @pgrep -a ydotoold > /dev/null && echo "  OK: ydotoold is running" || echo "  MISSING: Run 'just start-ydotool'"
