"""
Advertorch Integration - Attack Chaining and Composition
Combines multiple attack strategies for enhanced effectiveness
"""

import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Tuple, Callable, Optional
from pathlib import Path


class AdvertorchAttack:
    """Attack chaining and composition framework"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.attack_registry = {}
        self._register_default_attacks()

    def _register_default_attacks(self):
        """Register default attack functions"""
        self.attack_registry = {
            'noise': self._noise_attack,
            'blur': self._blur_attack,
            'compression': self._compression_attack,
            'rotate': self._rotation_attack,
            'crop': self._crop_attack,
        }

    def register_attack(self, name: str, attack_fn: Callable):
        """
        Register custom attack function

        Args:
            name: Attack name
            attack_fn: Function that takes (image/audio, params) and returns modified version
        """
        self.attack_registry[name] = attack_fn

    # ===== BASIC ATTACKS =====

    def _noise_attack(self, image: Image.Image, params: Dict) -> Image.Image:
        """Add Gaussian noise"""
        img_array = np.array(image).astype(np.float32)
        noise = np.random.randn(*img_array.shape) * params.get('std', 10)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def _blur_attack(self, image: Image.Image, params: Dict) -> Image.Image:
        """Apply Gaussian blur"""
        from PIL import ImageFilter
        radius = params.get('radius', 2)
        return image.filter(ImageFilter.GaussianBlur(radius))

    def _compression_attack(self, image: Image.Image, params: Dict) -> Image.Image:
        """JPEG compression attack"""
        from io import BytesIO
        quality = params.get('quality', 50)

        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def _rotation_attack(self, image: Image.Image, params: Dict) -> Image.Image:
        """Rotate image"""
        angle = params.get('angle', 5)
        return image.rotate(angle, fillcolor=(255, 255, 255))

    def _crop_attack(self, image: Image.Image, params: Dict) -> Image.Image:
        """Random crop and resize"""
        crop_pct = params.get('crop_pct', 0.9)
        w, h = image.size

        new_w = int(w * crop_pct)
        new_h = int(h * crop_pct)

        left = (w - new_w) // 2
        top = (h - new_h) // 2

        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.LANCZOS)

    # ===== ATTACK CHAINING =====

    def chain_attacks(self, image_path: str, attack_chain: List[Tuple[str, Dict]],
                     output_path: Optional[str] = None) -> Image.Image:
        """
        Execute a chain of attacks sequentially

        Args:
            image_path: Path to input image
            attack_chain: List of (attack_name, params) tuples
            output_path: Optional output path

        Returns:
            Final adversarial image
        """
        image = Image.open(image_path).convert('RGB')

        for attack_name, params in attack_chain:
            if attack_name not in self.attack_registry:
                print(f"Warning: Unknown attack '{attack_name}', skipping")
                continue

            attack_fn = self.attack_registry[attack_name]
            image = attack_fn(image, params)

        if output_path:
            image.save(output_path)

        return image

    def parallel_attacks(self, image_path: str, attacks: List[Tuple[str, Dict]],
                        output_dir: str = "media/parallel") -> Dict[str, str]:
        """
        Execute multiple attacks in parallel (generate all variants)

        Args:
            image_path: Path to input image
            attacks: List of (attack_name, params) tuples
            output_dir: Output directory

        Returns:
            Dictionary mapping attack name to output path
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = {}

        for attack_name, params in attacks:
            if attack_name not in self.attack_registry:
                continue

            image = Image.open(image_path).convert('RGB')
            attack_fn = self.attack_registry[attack_name]
            adversarial = attack_fn(image, params)

            output_path = f"{output_dir}/{attack_name}.png"
            adversarial.save(output_path)
            results[attack_name] = output_path

        return results

    def ensemble_attack(self, image_path: str, attacks: List[Tuple[str, Dict]],
                       blend_weights: Optional[List[float]] = None,
                       output_path: Optional[str] = None) -> Image.Image:
        """
        Ensemble attack - blend multiple attack results

        Args:
            image_path: Path to input image
            attacks: List of (attack_name, params) tuples
            blend_weights: Weights for blending (default: equal)
            output_path: Optional output path

        Returns:
            Blended adversarial image
        """
        if blend_weights is None:
            blend_weights = [1.0 / len(attacks)] * len(attacks)

        # Generate all variants
        variants = []
        for attack_name, params in attacks:
            if attack_name not in self.attack_registry:
                continue

            image = Image.open(image_path).convert('RGB')
            attack_fn = self.attack_registry[attack_name]
            adversarial = attack_fn(image, params)
            variants.append(np.array(adversarial).astype(np.float32))

        # Blend
        blended = np.zeros_like(variants[0])
        for variant, weight in zip(variants, blend_weights):
            blended += variant * weight

        blended = np.clip(blended, 0, 255).astype(np.uint8)
        result = Image.fromarray(blended)

        if output_path:
            result.save(output_path)

        return result

    def adaptive_attack(self, image_path: str, target_metric: str = 'perturbation',
                       max_iterations: int = 10) -> Tuple[Image.Image, List[str]]:
        """
        Adaptive attack - selects attacks based on effectiveness

        Args:
            image_path: Path to input image
            target_metric: Metric to optimize ('perturbation', 'quality', etc.)
            max_iterations: Maximum number of attack iterations

        Returns:
            Best adversarial image and attack sequence used
        """
        image = Image.open(image_path).convert('RGB')
        original = np.array(image)

        best_image = image
        best_score = 0
        attack_sequence = []

        for i in range(max_iterations):
            # Try each attack
            best_attack = None
            best_params = None
            best_variant = None
            best_iter_score = 0

            for attack_name in self.attack_registry.keys():
                # Generate with default params
                params = self._get_default_params(attack_name)
                attack_fn = self.attack_registry[attack_name]
                variant = attack_fn(best_image, params)

                # Evaluate
                score = self._evaluate_attack(original, np.array(variant), target_metric)

                if score > best_iter_score:
                    best_iter_score = score
                    best_attack = attack_name
                    best_params = params
                    best_variant = variant

            if best_attack is None:
                break

            best_image = best_variant
            best_score = best_iter_score
            attack_sequence.append(best_attack)

        return best_image, attack_sequence

    def _get_default_params(self, attack_name: str) -> Dict:
        """Get default parameters for attack"""
        defaults = {
            'noise': {'std': 10},
            'blur': {'radius': 2},
            'compression': {'quality': 50},
            'rotate': {'angle': 5},
            'crop': {'crop_pct': 0.9}
        }
        return defaults.get(attack_name, {})

    def _evaluate_attack(self, original: np.ndarray, adversarial: np.ndarray,
                        metric: str) -> float:
        """Evaluate attack effectiveness"""
        if metric == 'perturbation':
            # L2 distance
            diff = original.astype(np.float32) - adversarial.astype(np.float32)
            return np.sqrt(np.mean(diff ** 2))
        elif metric == 'quality':
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = np.mean((original - adversarial) ** 2)
            if mse == 0:
                return 100
            return 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            return 0.0

    def get_attack_strategies(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """
        Get predefined attack strategies

        Returns:
            Dictionary of strategy name to attack chain
        """
        return {
            'stealth': [
                ('noise', {'std': 5}),
                ('compression', {'quality': 85})
            ],
            'aggressive': [
                ('noise', {'std': 20}),
                ('blur', {'radius': 3}),
                ('rotate', {'angle': 10})
            ],
            'quality_degradation': [
                ('compression', {'quality': 30}),
                ('crop', {'crop_pct': 0.8})
            ],
            'geometric': [
                ('rotate', {'angle': 15}),
                ('crop', {'crop_pct': 0.85})
            ],
            'combined': [
                ('noise', {'std': 10}),
                ('blur', {'radius': 2}),
                ('compression', {'quality': 60}),
                ('rotate', {'angle': 3})
            ]
        }
