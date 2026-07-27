"""Non-blocking spoken warnings for Mendel Linux."""

import shutil
import subprocess
import time


SPEECH_NAMES = {
    "Bike": "bicycle",
    "Teraffic Barrel": "traffic barrel",
}


class AudioFeedback:
    def __init__(self, cooldown_seconds=2.5, speech_rate=155, enabled=True):
        self.cooldown_seconds = cooldown_seconds
        self.speech_rate = speech_rate
        self.enabled = enabled
        self.last_phrase = ""
        self.last_spoken_at = 0.0
        self.process = None
        self.command = shutil.which("espeak")

    @property
    def available(self):
        return self.command is not None

    def phrase_for(self, detection):
        class_name = SPEECH_NAMES.get(
            detection["class"],
            detection["class"].lower(),
        )
        return "{} {}, {}.".format(
            class_name,
            detection["distance"],
            detection["position"],
        )

    def speak(self, detection):
        if not self.enabled or detection is None:
            return False

        phrase = self.phrase_for(detection)
        now = time.monotonic()

        if now - self.last_spoken_at < self.cooldown_seconds:
            return False
        if phrase == self.last_phrase and now - self.last_spoken_at < 6.0:
            return False
        if self.process is not None and self.process.poll() is None:
            return False

        if not self.available:
            print("WARNING:", phrase)
            self.last_phrase = phrase
            self.last_spoken_at = now
            return False

        self.process = subprocess.Popen(
            [
                self.command,
                "-s",
                str(self.speech_rate),
                phrase,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.last_phrase = phrase
        self.last_spoken_at = now
        return True

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

