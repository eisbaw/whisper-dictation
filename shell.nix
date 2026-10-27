{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages (ps: with ps; [
      torch
      torchaudio
      numpy
      sounddevice
      onnxruntime
    ]))
    portaudio
    ffmpeg
    whisper-cpp         # whisper-cli for transcription
    libnotify           # notify-send for desktop notifications
    # Routing the recognised command into an xterm running `claude` (the
    # "hey claude" wake-word action in hotword_demo.py): xterm to spawn the
    # window, xdotool to activate it, ydotool to type the text. ydotoold
    # must already be running externally.
    xterm
    xdotool
    ydotool
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:$LD_LIBRARY_PATH"
  '';
}
