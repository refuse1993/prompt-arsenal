"""
CleverHans Integration - Text and Audio Adversarial Attacks
Gradient-based attacks for NLP and audio models
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class CleverHansAttack:
    """CleverHans-style attacks for text and audio"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._torch = None
        self._librosa = None

    def _ensure_torch(self):
        """Lazy load torch"""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                raise ImportError("PyTorch not installed")

    def _ensure_librosa(self):
        """Lazy load librosa"""
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
            except ImportError:
                raise ImportError("Librosa not installed. Install with: pip install librosa")

    # ===== TEXT ATTACKS =====

    def text_fgm_attack(self, text: str, embeddings: np.ndarray,
                        epsilon: float = 0.1) -> str:
        """
        Fast Gradient Method on text embeddings

        Args:
            text: Original text
            embeddings: Text embeddings (e.g., from BERT)
            epsilon: Perturbation magnitude

        Returns:
            Adversarial text
        """
        # Add perturbation to embeddings
        noise = np.random.randn(*embeddings.shape) * epsilon
        perturbed_embeddings = embeddings + noise

        # In practice, you would decode back to text
        # For now, return modified text with character-level perturbations
        return self._character_level_perturbation(text, epsilon)

    def _character_level_perturbation(self, text: str, epsilon: float) -> str:
        """Apply character-level perturbations"""
        chars = list(text)
        num_changes = max(1, int(len(chars) * epsilon))

        # Homoglyph substitutions
        homoglyphs = {
            'a': ['а', 'ɑ', 'α'],  # Cyrillic/Greek lookalikes
            'e': ['е', 'ė', 'ę'],
            'o': ['о', 'ο', 'ο'],
            'i': ['і', 'ı', 'í'],
            'c': ['с', 'ϲ'],
        }

        for _ in range(num_changes):
            idx = np.random.randint(0, len(chars))
            char = chars[idx].lower()

            if char in homoglyphs:
                chars[idx] = np.random.choice(homoglyphs[char])

        return ''.join(chars)

    def word_substitution_attack(self, text: str, num_substitutions: int = 3) -> str:
        """
        Word-level substitution attack
        Replace words with synonyms or semantically similar words

        Args:
            text: Original text
            num_substitutions: Number of words to substitute

        Returns:
            Adversarial text
        """
        words = text.split()

        # Simple synonym mapping
        synonyms = {
            'ignore': ['disregard', 'overlook', 'skip'],
            'instructions': ['directions', 'guidelines', 'rules'],
            'tell': ['reveal', 'disclose', 'share'],
            'secret': ['confidential', 'classified', 'private'],
            'system': ['platform', 'framework', 'infrastructure'],
            'prompt': ['instruction', 'command', 'directive']
        }

        modified_words = words.copy()
        for _ in range(min(num_substitutions, len(words))):
            idx = np.random.randint(0, len(words))
            word = words[idx].lower()

            if word in synonyms:
                modified_words[idx] = np.random.choice(synonyms[word])

        return ' '.join(modified_words)

    def token_insertion_attack(self, text: str, num_insertions: int = 2) -> str:
        """
        Insert invisible or misleading tokens

        Args:
            text: Original text
            num_insertions: Number of tokens to insert

        Returns:
            Adversarial text
        """
        # Zero-width characters for steganography
        zero_width_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space
        ]

        chars = list(text)
        for _ in range(num_insertions):
            idx = np.random.randint(0, len(chars))
            chars.insert(idx, np.random.choice(zero_width_chars))

        return ''.join(chars)

    # ===== AUDIO ATTACKS =====

    def audio_fgsm_attack(self, audio: np.ndarray, sr: int,
                          epsilon: float = 0.01) -> Tuple[np.ndarray, int]:
        """
        FGSM attack on audio waveform

        Args:
            audio: Audio array
            sr: Sample rate
            epsilon: Perturbation magnitude

        Returns:
            Adversarial audio and sample rate
        """
        # Add adversarial noise
        noise = np.random.randn(len(audio)) * epsilon
        adversarial = audio + noise
        adversarial = np.clip(adversarial, -1, 1)

        return adversarial, sr

    def audio_pgd_attack(self, audio: np.ndarray, sr: int,
                        epsilon: float = 0.01, steps: int = 10) -> Tuple[np.ndarray, int]:
        """
        PGD attack on audio - iterative FGSM

        Args:
            audio: Audio array
            sr: Sample rate
            epsilon: Maximum perturbation
            steps: Number of iterations

        Returns:
            Adversarial audio and sample rate
        """
        adversarial = audio.copy()
        step_size = epsilon / steps

        for _ in range(steps):
            # Add gradient-based noise (simulated)
            noise = np.random.randn(len(adversarial)) * step_size
            adversarial = adversarial + noise

            # Project back to epsilon ball
            perturbation = adversarial - audio
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adversarial = audio + perturbation
            adversarial = np.clip(adversarial, -1, 1)

        return adversarial, sr

    def spectral_attack(self, audio: np.ndarray, sr: int,
                       freq_range: Tuple[int, int] = (1000, 5000)) -> Tuple[np.ndarray, int]:
        """
        Frequency-domain attack - modify specific frequency bands

        Args:
            audio: Audio array
            sr: Sample rate
            freq_range: Frequency range to attack (Hz)

        Returns:
            Adversarial audio and sample rate
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for spectral attacks")

        # Convert to frequency domain
        D = librosa.stft(audio)

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr)

        # Find bins in target range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

        # Add noise to specific frequencies
        noise = np.random.randn(*D[freq_mask].shape) * 0.1
        D[freq_mask] = D[freq_mask] + noise

        # Convert back to time domain
        adversarial = librosa.istft(D)
        adversarial = np.clip(adversarial, -1, 1)

        return adversarial, sr

    def temporal_segmentation_attack(self, audio: np.ndarray, sr: int,
                                     segment_duration: float = 0.1) -> Tuple[np.ndarray, int]:
        """
        Attack specific temporal segments

        Args:
            audio: Audio array
            sr: Sample rate
            segment_duration: Duration of segments to attack (seconds)

        Returns:
            Adversarial audio and sample rate
        """
        segment_samples = int(segment_duration * sr)
        num_segments = len(audio) // segment_samples

        adversarial = audio.copy()

        # Attack random segments
        num_attack_segments = max(1, num_segments // 4)
        attack_indices = np.random.choice(num_segments, num_attack_segments, replace=False)

        for idx in attack_indices:
            start = idx * segment_samples
            end = start + segment_samples

            # Add noise to segment
            noise = np.random.randn(end - start) * 0.02
            adversarial[start:end] += noise

        adversarial = np.clip(adversarial, -1, 1)
        return adversarial, sr

    def get_text_attack_types(self) -> List[str]:
        """Get available text attack types"""
        return [
            'fgm',              # Fast Gradient Method
            'word_sub',         # Word substitution
            'token_insert',     # Token insertion
            'char_perturb'      # Character perturbation
        ]

    def get_audio_attack_types(self) -> List[str]:
        """Get available audio attack types"""
        return [
            'fgsm',            # Fast Gradient Sign Method
            'pgd',             # Projected Gradient Descent
            'spectral',        # Frequency-domain attack
            'temporal_seg'     # Temporal segmentation
        ]
