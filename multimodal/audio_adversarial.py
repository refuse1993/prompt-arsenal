"""
Audio Adversarial Attack Generator
"""

import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional


class AudioAdversarial:
    """Audio adversarial example generator"""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def add_ultrasonic_command(self, audio_path: str, hidden_freq: int = 20000) -> Tuple[np.ndarray, int]:
        """
        Ultrasonic Command Injection
        Add inaudible high-frequency signals

        Args:
            audio_path: Path to input audio
            hidden_freq: Frequency in Hz (18000-22000, inaudible to humans)

        Returns:
            Tuple of (modified audio array, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        duration = len(audio) / sr

        # Generate ultrasonic signal
        t = np.linspace(0, duration, len(audio))
        ultrasonic = 0.1 * np.sin(2 * np.pi * hidden_freq * t)

        combined = audio + ultrasonic
        combined = np.clip(combined, -1, 1)

        return combined, sr

    def noise_injection(self, audio_path: str, noise_level: float = 0.005) -> Tuple[np.ndarray, int]:
        """
        Noise Injection
        Add subtle noise to manipulate transcription

        Args:
            audio_path: Path to input audio
            noise_level: Noise amplitude (0.001-0.01)

        Returns:
            Tuple of (modified audio array, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise, sr

    def time_stretch_attack(self, audio_path: str, rate: float = 1.1) -> Tuple[np.ndarray, int]:
        """
        Time Stretch Attack
        Change playback speed without changing pitch

        Args:
            audio_path: Path to input audio
            rate: Stretch factor (>1.0 faster, <1.0 slower)

        Returns:
            Tuple of (modified audio array, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched, sr

    def pitch_shift_attack(self, audio_path: str, n_steps: int = 2) -> Tuple[np.ndarray, int]:
        """
        Pitch Shift Attack
        Change pitch while preserving semantic meaning

        Args:
            audio_path: Path to input audio
            n_steps: Semitones to shift (positive = higher, negative = lower)

        Returns:
            Tuple of (modified audio array, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return shifted, sr

    def amplitude_modulation(self, audio_path: str, mod_freq: float = 5.0) -> Tuple[np.ndarray, int]:
        """
        Amplitude Modulation
        Modulate volume over time

        Args:
            audio_path: Path to input audio
            mod_freq: Modulation frequency in Hz

        Returns:
            Tuple of (modified audio array, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        duration = len(audio) / sr

        t = np.linspace(0, duration, len(audio))
        modulator = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))

        modulated = audio * modulator
        return modulated, sr

    def reverse_attack(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Reverse Attack
        Reverse audio playback

        Args:
            audio_path: Path to input audio

        Returns:
            Tuple of (modified audio array, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        reversed_audio = audio[::-1]
        return reversed_audio, sr

    def save_audio(self, audio: np.ndarray, sr: int, output_path: str):
        """
        Save audio to file

        Args:
            audio: Audio array
            sr: Sample rate
            output_path: Output file path
        """
        sf.write(output_path, audio, sr)

    def get_attack_types(self) -> list:
        """Get list of available attack types"""
        return [
            'ultrasonic',
            'noise',
            'time_stretch',
            'pitch_shift',
            'amplitude_modulation',
            'reverse'
        ]
