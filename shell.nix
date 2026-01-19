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
    openai-whisper-cpp  # whisper-cli for transcription
    libnotify           # notify-send for desktop notifications
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:$LD_LIBRARY_PATH"
  '';
}
