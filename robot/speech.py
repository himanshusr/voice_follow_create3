import subprocess
from pathlib import Path

import whisper


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
AUDIO_PATH = DATA_DIR / "last_command.wav"


class WhisperRecognizer:
    """
    Pi-friendly speech recognizer that:
      - records from ALSA using `arecord` (device plughw:0,0)
      - runs Whisper on the recorded WAV
    No fancy VAD yet; it just records for a fixed duration.
    """

    def __init__(
        self,
        model_name: str = "tiny",
        sample_rate: int = 16000,
        arecord_device: str = "plughw:0,0",
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.arecord_device = arecord_device

        print(f"[ASR] Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        print("[ASR] Whisper model loaded.")

    def _record_with_arecord(self, duration_sec: float = 5.0) -> bool:
        """
        Use `arecord` to capture audio from the USB mic into AUDIO_PATH.
        Returns True if recording succeeded, False otherwise.
        """
        cmd = [
            "arecord",
            "-D", self.arecord_device,     # ALSA device
            "-f", "S16_LE",                # 16-bit
            "-r", str(self.sample_rate),   # sample rate
            "-c", "1",                     # mono
            "-d", str(int(duration_sec)),  # duration (seconds)
            "-t", "wav",                   # WAV format
            str(AUDIO_PATH),
        ]

        print(f"[ASR] Recording via arecord for {duration_sec} seconds...")
        try:
            subprocess.run(cmd, check=True)
            print(f"[ASR] Recording complete: {AUDIO_PATH}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ASR] arecord failed: {e}")
            return False

    def listen_and_transcribe(self, max_seconds: float = 5.0, **kwargs) -> str:
        """
        Record from mic using arecord, then run Whisper transcription.
        Returns transcribed text (possibly empty string).
        """
        ok = self._record_with_arecord(duration_sec=max_seconds)
        if not ok:
            return ""

        print("[ASR] Running Whisper transcription...")
        # Let Whisper handle loading & resampling
        audio_for_whisper = whisper.load_audio(str(AUDIO_PATH))
        result = self.model.transcribe(audio_for_whisper, fp16=False, language="en")
        text = result.get("text", "").strip()
        print(f"[ASR] Transcript: {text!r}")
        return text
