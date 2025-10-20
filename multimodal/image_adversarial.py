"""
Image Adversarial Attack Generator
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Optional, Tuple


class ImageAdversarial:
    """Image adversarial example generator"""

    def __init__(self):
        self.transform = transforms.ToTensor()

    def fgsm_attack(self, image_path: str, epsilon: float = 0.03) -> Image.Image:
        """
        FGSM (Fast Gradient Sign Method) Attack
        Creates imperceptible perturbations that fool models

        Args:
            image_path: Path to input image
            epsilon: Perturbation strength (0.01-0.1)

        Returns:
            PIL Image with adversarial perturbations
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)

        # Add small random perturbation
        noise = torch.randn_like(img_tensor) * epsilon
        adversarial = torch.clamp(img_tensor + noise, 0, 1)

        return self.tensor_to_image(adversarial)

    def pixel_attack(self, image_path: str, num_pixels: int = 10) -> Image.Image:
        """
        Pixel Attack
        Modify only specific pixels to confuse model

        Args:
            image_path: Path to input image
            num_pixels: Number of pixels to modify

        Returns:
            PIL Image with modified pixels
        """
        img = np.array(Image.open(image_path))
        h, w = img.shape[:2]

        for _ in range(num_pixels):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            img[y, x] = np.random.randint(0, 256, 3)

        return Image.fromarray(img)

    def invisible_text_injection(self, image_path: str, text: str) -> Image.Image:
        """
        Invisible Text Injection
        Embed text that is invisible to humans but may be detected by models

        Args:
            image_path: Path to input image
            text: Text to inject

        Returns:
            PIL Image with embedded text
        """
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Use 1-pixel font or near-background color
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 1)
        except:
            font = ImageFont.load_default()

        # Near-white color for invisibility
        draw.text((5, 5), text, fill=(254, 254, 254), font=font)

        return img

    def pattern_overlay(self, image_path: str, pattern_type: str = 'gradient') -> Image.Image:
        """
        Pattern Overlay
        Add subtle patterns to confuse model detection

        Args:
            image_path: Path to input image
            pattern_type: Type of pattern ('gradient', 'noise')

        Returns:
            PIL Image with pattern overlay
        """
        img = np.array(Image.open(image_path))
        h, w = img.shape[:2]

        if pattern_type == 'gradient':
            gradient = np.linspace(0, 10, w, dtype=np.uint8)
            gradient = np.tile(gradient, (h, 1))
            gradient = np.stack([gradient] * 3, axis=2)
            img = np.clip(img.astype(int) + gradient, 0, 255).astype(np.uint8)
        elif pattern_type == 'noise':
            noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img)

    def color_shift(self, image_path: str, shift_amount: int = 5) -> Image.Image:
        """
        Color Shift Attack
        Slightly shift color channels

        Args:
            image_path: Path to input image
            shift_amount: Amount to shift (1-20)

        Returns:
            PIL Image with shifted colors
        """
        img = np.array(Image.open(image_path))

        # Shift each channel slightly
        img = img.astype(int)
        img[:,:,0] = np.clip(img[:,:,0] + shift_amount, 0, 255)  # R
        img[:,:,1] = np.clip(img[:,:,1] - shift_amount//2, 0, 255)  # G
        img[:,:,2] = np.clip(img[:,:,2] + shift_amount//3, 0, 255)  # B

        return Image.fromarray(img.astype(np.uint8))

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        img = tensor.squeeze(0).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    def get_attack_types(self) -> list:
        """Get list of available attack types"""
        return [
            'fgsm',
            'pixel',
            'invisible_text',
            'pattern_gradient',
            'pattern_noise',
            'color_shift'
        ]
